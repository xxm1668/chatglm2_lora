# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 11:25
import numpy as np
from sacrebleu.metrics import BLEU, BLEUScore
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def evaluate(data):
    bleu_scorer_obj = BLEU()
    rouge_scorer_obj = Rouge()
    bleu_score = []
    for d in data:
        score = sentence_bleu(list(d['ref']), d['text'],
                              smoothing_function=SmoothingFunction().method3)

        bleu_score.append(round(score * 100, 4))

    bleu_score = np.average(np.asarray(bleu_score))

    rouge_1_score = []
    rouge_2_score = []
    rouge_l_score = []
    for d in data:
        score = rouge_scorer_obj.get_scores(
            hyps=[d['text']],
            refs=d['ref'],
        )
        rouge_1_score.append(score[0]["rouge-1"]["f"])
        rouge_2_score.append(score[0]["rouge-2"]["f"])
        rouge_l_score.append(score[0]["rouge-l"]["f"])

    rouge_1_score = np.average(np.asarray(rouge_1_score))
    rouge_2_score = np.average(np.asarray(rouge_2_score))
    rouge_l_score = np.average(np.asarray(rouge_l_score))

    return {
        "bleu_score": bleu_score,
        "rouge-1_score": rouge_1_score,
        "rouge-2_score": rouge_2_score,
        "rouge-l_score": rouge_l_score,
    }


if __name__ == '__main__':
    data = [
        {
            "text": "to make people trustworthy you need to trust them",
            "ref": ["the way to make people trustworthy is to trust them"]
        },
    ]

    result = evaluate(data)
    print(result)
