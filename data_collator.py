from tqdm import tqdm
import transformers
import torch

model_name = "/home/house365ai/xxm/chatglm2-6b"
max_seq_length = 1024
skip_over_length = True

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True)

config = transformers.AutoConfig.from_pretrained(
    model_name, trust_remote_code=True, device_map='auto')


def preprocess(example):
    context = example["context"]
    target = example["target"]

    context_ids = tokenizer.encode(
        context,
        max_length=max_seq_length,
        truncation=True)

    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)

    input_ids = context_ids + target_ids + [config.eos_token_id]

    return {"input_ids": input_ids, "context_len": len(context_ids), 'target_len': len(target_ids)}


def data_collator(features: list):
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

    input_ids = []
    labels_list = []

    for length, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        context_len = feature["context_len"]

        labels = (
                [-100] * (context_len - 1) + ids[(context_len - 1):] + [-100] * (longest - length)
        )  # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss

        ids = ids + [tokenizer.pad_token_id] * (longest - length)

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labels))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def get_data(ds_train, ds_val):
    ds_train_token = ds_train.map(preprocess).select_columns(['input_ids', 'context_len', 'target_len'])
    if skip_over_length:
        ds_train_token = ds_train_token.filter(
            lambda example: example["context_len"] < max_seq_length and example["target_len"] < max_seq_length)

    ds_val_token = ds_val.map(preprocess).select_columns(['input_ids', 'context_len', 'target_len'])
    if skip_over_length:
        ds_val_token = ds_val_token.filter(
            lambda example: example["context_len"] < max_seq_length and example["target_len"] < max_seq_length)
    dl_train = torch.utils.data.DataLoader(ds_train_token, num_workers=2, batch_size=4,
                                           pin_memory=True, shuffle=True,
                                           collate_fn=data_collator)
    dl_val = torch.utils.data.DataLoader(ds_val_token, num_workers=2, batch_size=4,
                                         pin_memory=True, shuffle=True,
                                         collate_fn=data_collator)
    return dl_train, dl_val
