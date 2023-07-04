import pandas as pd
import json
import datasets
from sklearn.model_selection import train_test_split


# 将上下文整理成与推理时候一致，参照model.chat中的源码~
# model.build_inputs??
def build_inputs(query, history):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
    prompt += "[Round {}]\n\n问：{} -> \n\n答：".format(len(history) + 1, query)
    return prompt


def split_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data.append(json.loads(line))
    dftrain, dftest = train_test_split(data, test_size=0.01, random_state=42)
    train = []
    for x in dftrain:
        tmp = {}
        tmp['context'] = build_inputs(x['instruction'], history=x['history'])
        tmp['target'] = x['output']
        train.append(tmp)

    test = []
    for x in dftest:
        tmp = {}
        tmp['context'] = build_inputs(x['instruction'], history=x['history'])
        tmp['target'] = x['output']
        test.append(tmp)

    ds_train = datasets.Dataset.from_list(train)
    ds_val = datasets.Dataset.from_list(test)

    return ds_train, ds_val
