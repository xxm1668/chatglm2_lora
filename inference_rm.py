from transformers import AutoTokenizer, AutoModel, AutoConfig, HfArgumentParser
import torch
import json
import random
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from peft import PeftModel
from trl import AutoModelForCausalLMWithValueHead


class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_reward_score(reward_model, reward_tokenizer, question, answer, device):
    """
    Get the reward score for a given question and answer pair.
    """
    inputs = reward_tokenizer(question, answer, return_tensors='pt').to(device)
    _, _, values = reward_model(**inputs)
    rewards = [reward for reward in values[-1].to(torch.float32)]
    reward_meter = AverageMeter()
    reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))
    score = reward_meter.avg
    return score


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    # Model arguments
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    reward_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "The reward model name"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    reward_baseline: Optional[float] = field(
        default=0.0, metadata={"help": "Baseline value that is subtracted from the reward"},
    )
    output_dir: Optional[str] = field(default="outputs-rl", metadata={"help": "The output directory"})
    use_v2: bool = field(default=True, metadata={"help": "Whether to use ChatGLM2-6b"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    # 加载模型model
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": True,
        "use_auth_token": None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        padding_side="left",
        **config_kwargs
    )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        **config_kwargs
    )

    model = AutoModel.from_pretrained(args.model_name_or_path, config=config, **config_kwargs)

    # Register auto class to save the custom code files.
    if hasattr(config, "auto_map") and "AutoConfig" in config.auto_map:
        config.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
        tokenizer.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoModel" in config.auto_map:
        model.__class__.register_for_auto_class()

    if args.use_v2:
        assert tokenizer.eos_token_id is not None, "Please update the *.json and *.py files of ChatGLM2-6B from HuggingFace."
        model.lm_head = model.transformer.output_layer
        output_embedding_base_layer = model.transformer
        output_embedding_layer_name = "output_layer"
    else:
        assert tokenizer.eos_token_id == 130005, "Please specify `use_v2` argument while using ChatGLM2-6B."
        output_embedding_base_layer = model
        output_embedding_layer_name = "lm_head"

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in ["layernorm"]):
            param.data = param.data.to(torch.float32)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # turn off when gradient checkpointing is enabled
    if hasattr(output_embedding_base_layer, output_embedding_layer_name):
        output_embedding_layer = getattr(output_embedding_base_layer, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(output_embedding_base_layer, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    model = PeftModel.from_pretrained(model, args.reward_model_name_or_path, is_trainable=True)
    model2 = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    model2.cuda()
    model2.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_filename = r'/home/xxm/fsdownload/chatglm2_lora/data/estate_qa.json'
    data = []
    with open(reward_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data.append(json.loads(line))

    data = random.sample(data, 100)
    for i, d in enumerate(tqdm(data)):
        # Compute reward score
        question = d['instruction']
        answer = d['output']
        score_outputs = [
            get_reward_score(model2, tokenizer, question, answer, device)
        ]
        print(score_outputs[0])


if __name__ == '__main__':
    main()
