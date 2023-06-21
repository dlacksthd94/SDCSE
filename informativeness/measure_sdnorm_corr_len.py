from re import A
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import os
from itertools import chain, product
import pandas as pd
from torch.nn.parallel import DataParallel
import json
from scipy.stats import spearmanr

PATH_DATA = os.path.join(os.getcwd(), 'data', 'wiki1m_for_simcse.txt')
with open(PATH_DATA, 'r') as f:
    list_text = f.readlines()
list_text = np.random.choice(list_text, size=100000, replace=False)
list_text = list_text.tolist()
len(list_text)

# BERT 모델 불러오기
with open('model_meta_data.json', 'r') as f:
    dict_model = json.load(f)
    list_encoder_to_remove = ['diffcse', 'promcse', 'scd']
    # list_encoder_to_remove = ['promcse', 'scd']
    for encoder in list_encoder_to_remove:
        dict_model.pop(encoder) 

INIT_DROPOUT = 0
BATCH_SIZE = 128
DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

columns = pd.MultiIndex.from_product([dict_model.keys(), dict_model['bert'].keys(), dict_model['bert']['bert'].keys()], names=['encoder', 'plm', 'size'])
list_max_len = [32, 64]
df = pd.DataFrame(index=list_max_len, columns=columns)
df.index.name = 'max_len'
encoder, plm, size = 'simcse', 'roberta', 'large'
for encoder in tqdm(dict_model):
    for plm in tqdm(dict_model[encoder], leave=False):
        for size in tqdm(dict_model[encoder][plm], leave=False):
            model_path = dict_model[encoder][plm][size]
            
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if torch.cuda.device_count() > 1:
                model = DataParallel(model)
                batch_size = BATCH_SIZE * torch.cuda.device_count()
            else:
                batch_size = int(BATCH_SIZE)
            _ = model.to(DEVICE)
            
            list_batch = [list_text[i:i+batch_size] for i in range(0, len(list_text), batch_size)]
            len(list_batch)

            batch = list_batch[0]
            def func(batch, max_len):
                tokenized_input = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                seq_len = tokenized_input['attention_mask'][:, 1:-1].sum(dim=1)
                tokenized_input = {k: v.to(DEVICE) for k, v in tokenized_input.items()}
                output = model(**tokenized_input, return_dict=True)
                output = output.last_hidden_state.cpu().detach()
                return output, seq_len

            for max_len in list_max_len:
                list_norm, list_seq_len, list_len, list_corr = [], [], [], []
                for batch in tqdm(list_batch, leave=False):
                    output, seq_len = func(batch, max_len)
                    norm = output[:, 0].norm(dim=-1)
                    list_norm.extend(norm.tolist())
                    list_seq_len.extend(seq_len.tolist())
                    list_corr.append(spearmanr(norm, seq_len).correlation)
                    list_len.extend([len(sent.split()) for sent in batch])
                # spearmanr(list_norm, list_seq_len).correlation
                corr = np.corrcoef(list_norm, list_seq_len)[0, 1]
                # spearmanr(list_norm, list_len).correlation
                # np.corrcoef(list_norm, list_len)[0, 1]
                # np.array([list_norm, list_seq_len]).T[:10]
                
                df.loc[max_len, (encoder, plm, size)] = corr
df = df.astype(float).round(4)
df.to_csv('../sdnorm_seq_len.csv')