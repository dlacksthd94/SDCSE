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

BATCH_SIZE = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 32
POOLER_METHOD = 'wop'

with open(os.path.join(os.getcwd(), "data", "wiki1m_for_simcse.txt"), "r") as f:
    list_text = f.readlines()
len(list_text)
list_text = random.sample(list_text, 20000)

list_encoder = ['simcse', 'diffcse', 'promcse', 'scd']
list_plm = ['bert', 'roberta']
list_size = ['base', 'large']
list_score = ['mean', 'std', '>0.2', '>0.4', '>0.6', 'iso', 'cos', 'uni', 'pdist', 'pdist_norm', 'norm_mean', 'norm_std']

with open('model_meta_data.json', 'r') as f:
    dict_model = json.load(f)

df = pd.DataFrame(columns=pd.MultiIndex.from_product([list_encoder, list_plm, list_size]), index=list_score)

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

for encoder in list_encoder:
    for plm in list_plm:
        for size in list_size:
            try:
                encoder, plm, size
                model_name = dict_model[encoder][plm][size]
                wrapper = SimCSE(model_name)
                
                tokenizer = wrapper.tokenizer
                model = wrapper.model
                _ = model.train()
                model.training
                if torch.cuda.device_count() > 1:
                    model = DataParallel(model)
                    BATCH_SIZE *= torch.cuda.device_count()
                _ = model.to(DEVICE)
                
                embedding_list = []
                with torch.no_grad():
                    total_batch = len(list_text) // BATCH_SIZE + (1 if len(list_text) % BATCH_SIZE > 0 else 0)
                    for batch_id in tqdm(range(total_batch)):
                        inputs = tokenizer(
                            list_text[batch_id*BATCH_SIZE:(batch_id+1)*BATCH_SIZE], 
                            padding=True, 
                            truncation=True, 
                            max_length=MAX_LEN, 
                            return_tensors="pt"
                        )
                        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                        outputs = model(**inputs, return_dict=True)
                        if POOLER_METHOD == "wp":
                            embeddings = outputs.pooler_output
                        elif POOLER_METHOD == "wop":
                            embeddings = outputs.last_hidden_state[:, 0]
                        else:
                            raise NotImplementedError
                        # if normalize_to_unit:
                        #     embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                        embedding_list.append(embeddings.cpu())
                embeddings = torch.cat(embedding_list, 0)
                
                embeddings_mean = embeddings.mean(dim=0)
                embeddings_mean_total = embeddings_mean.mean().item()
                over_2 = sum(abs(embeddings_mean) >= 0.2).item()
                over_4 = sum(abs(embeddings_mean) >= 0.4).item()
                over_6 = sum(abs(embeddings_mean) >= 0.6).item()
                
                embeddings_std = embeddings.std(dim=0)
                embeddings_std_total = embeddings_std.mean().item()
                
                arr_cos = cosine_similarity(embeddings)
                arr_cos_mean = np.ma.masked_where(arr_cos == 0, arr_cos).mean()
                
                embeddings = embeddings.double()
                uniformity = uniform_loss(embeddings).item()
                
                norm_mean = embeddings.norm(dim=-1).mean().item()
                norm_std = embeddings.norm(dim=-1).std().item()
                
                pdist = torch.pdist(embeddings, p=2).mean().item()
                pdist_norm = pdist / norm_mean
                
                df.loc[:, (encoder, plm, size)] = [embeddings_mean_total, embeddings_std_total, over_2, over_4, over_6, np.nan, arr_cos_mean, uniformity, pdist, pdist_norm, norm_mean, norm_std]
            except KeyError as e:
                print(e)
                pass
df = df.dropna(axis=1, how='all')
df = df.round(4)
df.to_csv('df_isotropy.csv')
pd.read_csv('df_isotropy.csv', header=[0,1,2], index_col=0)