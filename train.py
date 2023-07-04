from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from torchkeras import KerasModel
from model import StepRunner
from data_processon import split_data
from data_collator import get_data


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


model = AutoModel.from_pretrained("/home/house365ai/xxm/chatglm2-6b",
                                  trust_remote_code=True,
                                  device_map='auto')

model.supports_gradient_checkpointing = True  # 节约cuda
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
# model.lm_head = CastOutputToFloat(model.lm_head)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.05,
)
model = model.half()
model = get_peft_model(model, peft_config)
model.is_parallelizable = True
model.model_parallel = True
model.print_trainable_parameters()
KerasModel.StepRunner = StepRunner
KerasModel.save_ckpt = StepRunner.save_ckpt
KerasModel.load_ckpt = StepRunner.load_ckpt
keras_model = KerasModel(model, loss_fn=None,
                         optimizer=torch.optim.AdamW(model.parameters(), lr=5e-4))
filename = r'/home/house365ai/xxm/chatglm2_lora/data/estate_qa.json'
ds_train, ds_val = split_data(filename)
dl_train, dl_val = get_data(ds_train, ds_val)
ckpt_path = '/home/house365ai/xxm/chatglm2_lora/output'
keras_model.fit(train_data=dl_train,
                val_data=dl_val,
                epochs=50, patience=5,
                monitor='val_loss', mode='min',
                ckpt_path=ckpt_path,
                mixed_precision='fp16'
                )
