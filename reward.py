from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME
import torch
from typing import Any, List, Union, Optional, Dict
import os
from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer, PreTrainedTokenizerBase, \
    TrainingArguments, \
    HfArgumentParser
from data_processon import split_reward_data
from data_collator import preprocess_reward_function
from dataclasses import dataclass, field
from trl import AutoModelForCausalLMWithValueHead
from sklearn.metrics import mean_squared_error, mean_absolute_error
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


@dataclass
class RewardDataCollatorWithPadding:
    """We need to define a special data collator that batches the data in our chosen vs rejected format"""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        for feature in features:
            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        return batch


@dataclass
class PeftArguments(TrainingArguments):
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    use_v2: bool = field(default=True, metadata={"help": "Whether to use ChatGLM2-6b"})
    checkpoint_dir: Optional[str] = field(default=None, metadata={"help": "lora path of model ChatGLM2-6b"})
    model_path: Optional[str] = field(default=None, metadata={"help": "path of model ChatGLM2-6b"})
    reward_filename: Optional[str] = field(default=None, metadata={"help": "path of reward filename"})


class RewardTrainer(Trainer):
    """
    Trainer for reward models
        Define how to compute the reward loss. Use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"],
                               attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_rejected = model(input_ids=inputs["input_ids_rejected"],
                                 attention_mask=inputs["attention_mask_rejected"])[0]
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Prepare inputs for chosen and rejected separately
        device = model.device

        inputs_chosen = {
            "input_ids": inputs["input_ids_chosen"].to(device),
            "attention_mask": inputs["attention_mask_chosen"].to(device),
        }
        outputs_chosen = model(**inputs_chosen)
        rewards_chosen = outputs_chosen.logits.detach()

        inputs_rejected = {
            "input_ids": inputs["input_ids_rejected"].to(device),
            "attention_mask": inputs["attention_mask_rejected"].to(device),
        }
        outputs_rejected = model(**inputs_rejected)
        rewards_rejected = outputs_rejected.logits.detach()

        # Keep the compute_loss method
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, rewards_chosen, rewards_rejected)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


class CastOutputToFloat(torch.nn.Sequential):
    """Cast the output of the model to float"""

    def forward(self, x):
        return super().forward(x).to(torch.float32)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Here, predictions is rewards_chosen and rewards_rejected.
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    # MSE
    mse = mean_squared_error(labels, preds)
    # MAE
    mae = mean_absolute_error(labels, preds)

    return {"mse": mse, "mae": mae}


def save_model(output_dir, model, tokenizer, args):
    """Save the model and the tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    print(checkpoint_dir)
    valuehead_file = os.path.join(checkpoint_dir, "value_head.bin")
    if not os.path.exists(valuehead_file):
        print("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    valuehead_state_dict = torch.load(valuehead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])
    model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
    model.register_buffer("default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"]))
    model.register_buffer("default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"]))
    return True


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param))


def main():
    # 加载超参数
    parser = HfArgumentParser(PeftArguments)
    training_args = parser.parse_args_into_dataclasses()[0]
    # 加载模型model
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": True,
        "use_auth_token": None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_path,
        use_fast=True,
        padding_side="left",
        **config_kwargs
    )

    config = AutoConfig.from_pretrained(
        training_args.model_path,
        **config_kwargs
    )

    model = AutoModel.from_pretrained(training_args.model_path, config=config, **config_kwargs)

    if hasattr(config, "auto_map") and "AutoConfig" in config.auto_map:
        config.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
        tokenizer.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoModel" in config.auto_map:
        model.__class__.register_for_auto_class()

    if training_args.use_v2:
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

    # load lora params
    lastest_checkpoint = training_args.checkpoint_dir

    if lastest_checkpoint is not None:  # resume lora training
        model = PeftModel.from_pretrained(model, lastest_checkpoint, is_trainable=True)

    if lastest_checkpoint is None:  # create new lora weights while training
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # we should regard ChatGLM as a causal LM
            inference_mode=False,
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
        )
        model = get_peft_model(model, lora_config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    # model.v_head.load_state_dict({
    #     "summary.weight": getattr(model, "reward_head_weight"),
    #     "summary.bias": getattr(model, "reward_head_bias")
    # })
    print_trainable_params(model)

    # 加载reward数据
    ds_train, ds_val = split_reward_data(training_args.reward_filename)

    tokenized_dataset = ds_train.shuffle().map(
        preprocess_reward_function,
        batched=True,
        num_proc=1,
        remove_columns=ds_train.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    train_dataset = tokenized_dataset.filter(
        lambda x: x is not None and 0 < len(x['input_ids_rejected']) <= 512 and 0 < len(
            x['input_ids_chosen']) <= 512
    )

    tokenized_dataset = ds_val.shuffle().map(
        preprocess_reward_function,
        batched=True,
        num_proc=1,
        remove_columns=ds_val.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    val_dataset = tokenized_dataset.filter(
        lambda x: x is not None and 0 < len(x['input_ids_rejected']) <= 512 and 0 < len(
            x['input_ids_chosen']) <= 512
    )
    # 增加不可忽略的字段名
    training_args.label_names = ['input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected',
                                 'attention_mask_rejected']

    print(training_args)
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=512, padding="max_length"
        ),

    )

    # training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(checkpoint)

    # saving
    metrics = train_result.metrics
    print(f"Training metrics: {metrics}")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(f"Saving model checkpoint to {training_args.output_dir}")
    save_model(training_args.output_dir, model, tokenizer, training_args)
    print('---------finsh--------')


if __name__ == '__main__':
    main()
