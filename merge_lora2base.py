# -*- coding: utf-8 -*-
import argparse

import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModel,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForSequenceClassification,
)

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model_name_or_path', default=None, required=True, type=str,
                        help="Base model name or path")
    parser.add_argument('--peft_model_path', default=None, required=True, type=str,
                        help="Please specify LoRA model to be merged.")
    parser.add_argument('--output_dir', default='./merged', type=str)

    args = parser.parse_args()
    print(args)
    base_model_path = args.base_model_name_or_path
    peft_model_path = args.peft_model_path
    output_dir = args.output_dir

    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {peft_model_path}")
    peft_config = PeftConfig.from_pretrained(peft_model_path)

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if peft_config.task_type == "SEQ_CLS":
        print("Loading LoRA for sequence classification model")
        if args.model_type == "chatglm":
            raise ValueError("chatglm does not support sequence classification")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=1,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        print("Loading LoRA for causal language model")
        base_model = model_class.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
    tokenizer = tokenizer_class.from_pretrained(base_model_path, trust_remote_code=True)
    base_model_token_size = base_model.get_input_embeddings().weight.size(0)
    if base_model_token_size != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Resize vocabulary size {base_model_token_size} to {len(tokenizer)}")

    lora_model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    lora_model.eval()
    print(f"Merging with merge_and_unload...")
    base_model = lora_model.merge_and_unload()

    print("Saving to Hugging Face format...")
    tokenizer.save_pretrained(output_dir)
    base_model.save_pretrained(output_dir)
    print(f"Done! model saved to {output_dir}")


if __name__ == '__main__':
    main()
