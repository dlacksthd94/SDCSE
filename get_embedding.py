import torch
from torch import nn
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import joblib as jl
from tqdm import tqdm
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

BATCH_SIZE = 64 # 32 or 64
PERTURBATION = True

if not os.path.exists(f'wiki1m_for_simcse_tokenized_{BATCH_SIZE}.pickle'):
    # load dataset
    with open('../SimCSE/wiki1m_for_simcse.txt') as f:
        list_text = f.readlines()

    # examples
    for text in list_text[:10]:
        print(text)
        
    # preprocessing
    for i in tqdm(range(len(list_text))):
        list_text[i] = list_text[i].strip()

    # make batch to load on GPU
    list_batch = []
    for i in tqdm(range(0, 1000000, BATCH_SIZE)):
        batch = tokenizer(text=list_text[i:i+BATCH_SIZE], padding=True, truncation=True, return_tensors="pt", verbose=True)
        list_batch.append(batch)

    with open(f'wiki1m_for_simcse_tokenized_{BATCH_SIZE}.pickle', 'wb') as f:
        pickle.dump(list_batch, f)
else:
    with open(f'wiki1m_for_simcse_tokenized_{BATCH_SIZE}.pickle', 'rb') as f:
        list_batch = pickle.load(f)

# if PERTURBATION:
#     batch['input_ids'][0]
#     batch['attention_mask'][0]
#     tokenizer.unk_token_id
#     tokenizer('for ')

for i in tqdm(range(len(list_batch))):
    _ = list_batch[i].to('cuda')

# get embeddings
_ = model.to('cuda')
list_embeddings = []
with torch.no_grad():
    for batch in tqdm(list_batch):
        embeddings_temp = model(**batch, output_hidden_states=True, return_dict=True).pooler_output
        list_embeddings.append(embeddings_temp)

embeddings = torch.stack(list_embeddings).reshape(-1, 768).detach().cpu().numpy()

with open(f'wiki1m_for_simcse_embedding.pickle', 'wb') as f:
    pickle.dump(embeddings.tolist(), f)