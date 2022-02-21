#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : nli_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/26 14:16
@version: 1.0
@desc  : 
"""

import json
import os
import pandas as pd
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class NLIDataset(Dataset):

    def __init__(self, csv_path, model_path, max_length: int = 128):
        super().__init__()
        self.max_length = max_length
        self.label_map = {"contradiction": 0, 'neutral': 1, "entailment": 2}
        self.result = pd.read_csv(csv_path) # dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        sentence_1 = self.result['premise'][idx]
        sentence_2 = self.result['hypothesis'][idx]
        label = self.label_map[self.result['label'][idx]]
        sentence_1_input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=False)
        sentence_2_input_ids = self.tokenizer.encode(sentence_2, add_special_tokens=False)
        input_ids = sentence_1_input_ids + [2] + sentence_2_input_ids
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
        # convert list to tensor
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2])
        label = torch.LongTensor([label])
        return input_ids, label, length
