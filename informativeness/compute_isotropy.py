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

BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 32
POOLER_METHOD = 'wp'
NUM_DATA = 100000

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

df = pd.DataFrame(columns=pd.MultiIndex.from_product([dict.keys(), list_plm, list_size]), index=list_score)

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

encoder, plm, size = 'simcse', 'roberta', 'large'
for encoder in tqdm(dict_model):
    for plm in tqdm(dict_model[encoder], leave=False):
        for size in tqdm(dict_model[encoder][plm], leave=False):
            try:
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
                
                embeddings_mean = embeddings.mean(dim=0)
                embeddings_mean_total = embeddings_mean.mean().item()
                embeddings_mean_std = embeddings_mean.std().item()
                over_05 = sum(abs(embeddings_mean) >= 0.5).item()
                over_10 = sum(abs(embeddings_mean) >= 1).item()
                
                embeddings_std = embeddings.std(dim=0)
                embeddings_std_total = embeddings_std.mean().item()
                
                # # embeddings = embeddings.double()
                # uniformity = uniform_loss(embeddings).item()
                uniformity = 0
                
                norm_mean = embeddings.norm(dim=-1).mean().item()
                norm_std = embeddings.norm(dim=-1).std().item()
                
                # pdist = torch.pdist(embeddings, p=2).mean().item()
                # pdist_norm = pdist / norm_mean
                pdist = 0
                pdist_norm = 0
                
                list_index = [random.choices(range(NUM_DATA), k=2) for i in range(NUM_DATA)]
                sampled_pair = torch.stack([embeddings[[i, j]] for i, j in list_index])
                arr_cos = np.array([cosine_similarity(x[0:1], x[1:2])[0][0] for x in tqdm(sampled_pair, leave=False)])
                # arr_cos_mean = np.ma.masked_where(arr_cos == 0, arr_cos).mean()
                arr_cos_mean = arr_cos.mean()
                
                df.loc[:, (encoder, plm, size)] = [embeddings_mean_total, embeddings_mean_std, embeddings_std_total, over_05, over_10, np.nan, arr_cos_mean, uniformity, pdist, pdist_norm, norm_mean, norm_std]
            except KeyError as e:
                print(e)
                pass
df = df.dropna(axis=1, how='all')
df = df.round(4)
df.to_csv('df_isotropy.csv')
pd.read_csv('df_isotropy.csv', header=[0,1,2], index_col=0)