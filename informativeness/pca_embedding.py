from simcse import SimCSE
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.nn.parallel import DataParallel
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import pandas as pd
import os
import itertools
import pickle
import random
import pandas as pd
import json
from transformers import BertModel, AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import copy
import matplotlib.pyplot as plt

BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 32
POOLER_METHOD = 'wp'
NUM_DATA = 10000

with open(os.path.join(os.getcwd(), "data", "wiki1m_for_simcse.txt"), "r") as f:
    list_text = f.readlines()
len(list_text)
list_text = random.sample(list_text, NUM_DATA)

list_plm = ['bert', 'roberta']
list_size = ['base', 'large']
list_score = ['mean', 'mean_std', 'std', '>0.5', '>1.0', 'iso', 'cos', 'uni', 'pdist', 'pdist_norm', 'norm_mean', 'norm_std']

with open('model_meta_data.json', 'r') as f:
    dict_model = json.load(f)
    list_encoder_to_remove = ['diffcse', 'promcse', 'scd']
    # list_encoder_to_remove = ['promcse', 'scd']
    for encoder in list_encoder_to_remove:
        dict_model.pop(encoder) 

df = pd.DataFrame(columns=pd.MultiIndex.from_product([dict_model.keys(), list_plm, list_size]), index=list_score)

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

encoder, plm, size = 'simcse', 'roberta', 'large'
dict_result = copy.deepcopy(dict_model)
for encoder in tqdm(dict_model):
    for plm in tqdm(dict_model[encoder], leave=False):
        for size in tqdm(dict_model[encoder][plm], leave=False):
            encoder, plm, size
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

            def func(batch):
                tokenized_input = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                tokenized_input = {k: v.to(DEVICE) for k, v in tokenized_input.items()}
                output = model(**tokenized_input, return_dict=True)
                if POOLER_METHOD == "ap":
                    output = output.pooler_output
                elif POOLER_METHOD == "wp":
                    output = output.last_hidden_state[:, 0]
                return output.cpu().detach()

            embedding_list = []
            for batch in tqdm(list_batch, leave=False):
                output = func(batch)
                embedding_list.append(output)
            embeddings = torch.cat(embedding_list, 0)

            # PCA 모델 인스턴스화
            k = 2
            pca = PCA(n_components=k)  # k는 원하는 주성분 수를 나타냅니다.

            # 데이터에 PCA 모델 적합
            pca.fit(embeddings)

            # 변환된 데이터 가져오기
            transformed_data = pca.transform(embeddings)

            dict_result[encoder][plm][size] = transformed_data

fig, ax = plt.subplots(2, 4, figsize=(12, 6))
for i, encoder in enumerate(dict_model):
    for j, plm in enumerate(dict_model[encoder]):
        for k, size in enumerate(dict_model[encoder][plm]):
            x = dict_result[encoder][plm][size]
            _ = ax[i][j * 2 + k].scatter(x[:, 0], x[:, 1], s=1, alpha=0.1)
            _ = ax[i][j * 2 + k].axhline(0, c='red', linewidth=0.5)
            _ = ax[i][j * 2 + k].axvline(0, c='red', linewidth=0.5)
            _ = ax[i][j * 2 + k].set_title(f'{encoder}-{plm}-{size}')
plt.tight_layout()
plt.savefig('pca_embedding.png')
plt.clf()