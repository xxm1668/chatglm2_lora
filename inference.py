from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import json
import random
from tqdm import tqdm

model_path = '/home/house365ai/xxm/chatglm2-6b'
ckpt_path = '/home/house365ai/xxm/chatglm2_lora/output/estate_qa1'
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

target_filename = r'/home/house365ai/xxm/chatglm2_lora/data/prediction.json'
target_w = open(target_filename, 'a+', encoding='utf-8')
data = random.sample(data, 500)
for i, d in enumerate(tqdm(data)):
    response = predict(d)
    tmp = {}
    print('query：', d['instruction'])
    print('response：', response)
    print('------------')
    tmp['ori_answer'] = d['output']
    tmp['pre_answer'] = response
    target_w.write(json.dumps(tmp, ensure_ascii=False) + '\n')
