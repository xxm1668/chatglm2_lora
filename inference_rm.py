from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser
import torch
import json
import random
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional


def get_reward_score(reward_model, reward_tokenizer, question, answer, device):
    """
    Get the reward score for a given question and answer pair.
    """
    inputs = reward_tokenizer(question, answer, return_tensors='pt').to(device)
    score = reward_model(**inputs).logits[0].cpu().detach()

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


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer_name_or_path = args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = args.model_name_or_path

    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name_or_path,
        load_in_8bit=args.load_in_8bit,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
    )
    reward_model.to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_name_or_path, **tokenizer_kwargs
    )

    reward_filename = r''
    data = []
    with open(reward_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data.append(json.loads(line))

    data = random.sample(data, 100)
    for d in enumerate(tqdm(data)):
        # Compute reward score
        score_outputs = [
            get_reward_score(reward_model, reward_tokenizer, q, r, device) for q, r in
            zip(d["instruction"], d["output"])
        ]
    rewards = [torch.tensor(float(score) - args.reward_baseline) for score in score_outputs]
    print(rewards)
    print('---------------')
