# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 11:25
import json

import numpy as np
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba

rouge = Rouge()


def compute_metrics(decoded_preds, decoded_labels):
    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            if pred:
                hypothesis = list(jieba.cut(str(pred)))
                if len(hypothesis) == 0:
                    hypothesis = ['*****']
            else:
                hypothesis = ['*****']
            if label:
                reference = list(jieba.cut(str(label)))
                if len(reference) == 0:
                    reference = ['*****']
            else:
                reference = ['*****']

            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))
        except Exception as e:
            print(e)
            print(pred)
            print(label)

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict


if __name__ == '__main__':
    ori_answer = []
    pre_answer = []
    filename = r'/Users/haojingkun/PycharmProjects/chatglm2_lora/data/prediction.json'
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_json = json.loads(line)
            ori_answer.append(data_json['ori_answer'])
            pre_answer.append(data_json['pre_answer'])

    result = compute_metrics(ori_answer, pre_answer)
    print(json.dumps(result, ensure_ascii=False, indent=4))
