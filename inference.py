from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import json
import random
from tqdm import tqdm

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


def predict(example):
    response, history = model.chat(tokenizer, f"{example['instruction']} -> ", history=example['history'],
                                   temperature=0.01, top_p=0.95, repetition_penalty=1.5)
    return response


filename = r'/home/house365ai/xxm/chatglm2_lora/data/estate_qa.json'
data = []
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        data.append(json.loads(line))

data = random.sample(data, 100)
for i, d in enumerate(tqdm(data)):
    print('query：', d['instruction'])
    print('response：', predict(d))
    print('------------')
