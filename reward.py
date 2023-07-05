from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME
import torch
from typing import Any, List, Union, Optional, Dict
import os
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, TrainingArguments, HfArgumentParser
from peft import PeftModel
from data_processon import split_reward_data
from data_collator import preprocess_reward_function
from dataclasses import dataclass, field
from sklearn.metrics import mean_squared_error, mean_absolute_error


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


def main():
    # 加载超参数
    parser = HfArgumentParser((PeftArguments,))
    training_args = parser.parse_args_into_dataclasses()
    # 加载模型model
    model_path = '/home/house365ai/xxm/chatglm2-6b'
    ckpt_path = '/home/house365ai/xxm/chatglm2_lora/output'
    model = AutoModel.from_pretrained(model_path,
                                      trust_remote_code=True,
                                      device_map='auto')
    model = model.half()
    model = PeftModel.from_pretrained(model, ckpt_path)
    model = model.merge_and_unload()  # 合并lora权重
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              device_map='auto')

    # 加载reward数据
    reward_filename = r'/home/house365ai/xxm/chatglm2_lora/data/estate_qa_reward.json'
    ds_train, ds_val = split_reward_data(reward_filename)
    tokenized_dataset = ds_train.shuffle().map(
        preprocess_reward_function,
        batched=True,
        num_proc=2,
        remove_columns=ds_train.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    train_dataset = tokenized_dataset.filter(
        lambda x: 0 < len(x['input_ids_rejected']) <= 1024 and 0 < len(
            x['input_ids_chosen']) <= 1024
    )

    tokenized_dataset = ds_train.shuffle().map(
        preprocess_reward_function,
        batched=True,
        num_proc=2,
        remove_columns=ds_val.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    val_dataset = tokenized_dataset.filter(
        lambda x: 0 < len(x['input_ids_rejected']) <= 1024 and 0 < len(
            x['input_ids_chosen']) <= 1024
    )

    # 初始化reward model模型
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=1024, padding="max_length"
        ),
    )

    # training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # saving
    metrics = train_result.metrics
    print(f"Training metrics: {metrics}")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(f"Saving model checkpoint to {training_args.output_dir}")
    save_model(training_args.output_dir, model, tokenizer, training_args)
    print('---------finsh--------')
