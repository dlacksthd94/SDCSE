import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, AutoModel
from scipy.stats import spearmanr
from tqdm import tqdm
import numpy as np
import os

# BERT 모델 불러오기
model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased')

PATH_DATA = os.path.join(os.getcwd(), 'data', 'wiki1m_for_simcse.txt')
with open(PATH_DATA, 'r') as f:
    list_text = f.readlines()
len(list_text)

arr_seq_len = np.array([len(x) for x in tokenizer(list_text)['input_ids']])
arr_seq_len.mean()
sum(arr_seq_len > 32)
sum(arr_seq_len > 64)
sum(arr_seq_len > 128)
# tokenizer(list_text, return_tensors='pt', padding=True, truncation=True, max_length=32)