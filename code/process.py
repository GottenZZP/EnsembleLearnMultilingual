import numpy as np
import pandas as pd
import time
import os
import re
import torch
import warnings
import json
from transformers import AlbertTokenizer
from torch.utils.data import DataLoader
from transformers import logging
from sklearn.model_selection import train_test_split
from sklearn import datasets
from transformers.utils.notebook import format_time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class InputDataSet:

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data['text'][index])
        labels = torch.tensor(self.data['label'][index], dtype=torch.long)

        output = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids, token_type_ids, attention_mask = output.values()
        # input_ids, attention_mask = output.values()
        input_ids = input_ids.squeeze(dim=0)
        attention_mask = attention_mask.squeeze(dim=0)
        # token_type_ids = token_type_ids.squeeze(dim=0)
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 'token_type_ids': token_type_ids,
            'labels': labels
        }


class TestInput:

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data['text'][index])

        output = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        # distilbert
        input_ids, attention_mask = output.values()
        # input_ids, token_type_ids, attention_mask = output.values()
        # input_ids, attention_mask, token_type_ids = output.values()
        input_ids = input_ids.squeeze(dim=0)
        attention_mask = attention_mask.squeeze(dim=0)
        # token_type_ids = token_type_ids.squeeze(dim=0)
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 'token_type_ids': token_type_ids,
        }


def my_dataset(train_path, tokenizer, max_len=512, test_path=None):
    train_data = pd.read_csv(train_path)
    train_dataset = InputDataSet(train_data, tokenizer, max_len)

    test_data = None
    test_dataset = None
    if test_path is not None:
        test_data = pd.read_csv(test_path)
        test_dataset = InputDataSet(test_data, tokenizer, max_len)

    return train_dataset, test_dataset


def change_dir_name(dir_path):
    os.chdir(dir_path)
    file_name_list = os.listdir(dir_path)
    for name in file_name_list:
        # old_name_path, extend_name = os.path.splitext(name)
        new_name = name.split('-')[0] + '_t-' + name.split('-')[1]
        os.rename(name, new_name)
        print(name, new_name)


def read_file(data_dir):
    """读取文件"""
    data = pd.read_csv(data_dir)
    # 将缺省数据Nan填充为空字符
    data['text'] = data['text'].fillna('')
    return data


def process_text(data):
    """将数据中的垃圾字符去除"""
    text = list(data['text'])
    for i in range(len(text)):
        text[i] = text[i].strip().replace('XXXX', '')
    data['text'] = text
    return data


def get_label(path):
    df = pd.read_csv(path)
    label = df['label']
    df2 = pd.read_csv("../dataset/data/idx.csv")
    idx, label_i = df2['idx'], df2['label']

    label_to_idx = dict()
    for l, i in zip(label_i, idx):
        label_to_idx[i] = l

    out = []
    for l in label:
        out.append([label_to_idx[l], l])

    df3 = pd.DataFrame(out, columns=["label", "idx"])
    df3.to_csv("../dataset/data/val3_label.csv", sep=',', index=False)


if __name__ == '__main__':
    # change_dir_name('D:\python_code\paper_extend\exp_data\\loss')
    get_label("D:/python_code/paper_extend/dataset/data/val3.csv")
