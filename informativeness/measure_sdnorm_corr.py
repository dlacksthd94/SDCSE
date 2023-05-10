from asyncio.subprocess import DEVNULL
from multiprocessing import Barrier
from simcse import SimCSE
# from diffcse import DiffCSE
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

def get_norm(tokenized_inputs_batch):
    global DEVICE
    tokenized_inputs_batch = {k: v.to(DEVICE) for k, v in tokenized_inputs_batch.items()}
    outputs = model(**tokenized_inputs_batch, return_dict=True)
    embeddings = outputs.last_hidden_state[:, 0]
    return embeddings.norm(dim=-1).cpu().detach()#.numpy().round(4)

def calculate_corr(tokenized_inputs_interleaved, num_dup):
    global BATCH_SIZE
    num_sent_in_batch = BATCH_SIZE // num_dup
    adjusted_batch_size = num_sent_in_batch * num_dup
    
    # idx_to_select = [x for x in range(tokenized_inputs_interleaved['input_ids'].shape[0]) if x % (n_perturbate_max + 1) in range(n_perturbate + 1)]
    # tokenized_inputs_selected = {id_type: tensor[idx_to_select] for id_type, tensor in tokenized_inputs_interleaved.items()}

    total_len = tokenized_inputs_interleaved['input_ids'].shape[0]
    list_result = []
    for i in tqdm(range(0, total_len, adjusted_batch_size), leave=False):
        tokenized_inputs_batch = {id_type: tensor[i:i + adjusted_batch_size] for id_type, tensor in tokenized_inputs_interleaved.items()}        
        result = get_norm(tokenized_inputs_batch)
        result = result.reshape(-1, num_dup)
        list_result.append(result)
    tsr_result = torch.cat(list_result)
    
    list_corr = []    
    for i in range(1, num_dup):
        x = torch.argsort(tsr_result[:, :i + 1]).float()
        y = torch.Tensor([range(i, -1, -1) for _ in range(tsr_result.shape[0])])
        spearman_corr = 1 - (x - y).pow(2).sum(dim=1).mul(6).div((i + 1) * ((i + 1) ** 2 - 1)) # same as [spearmanr(*pair).correlation for pair in zip(x, y)]
        list_corr.append(spearman_corr.mean().item())

    return list_corr, tsr_result

def experiment(num_sent, n_perturbate_max=9, step_max=9):
    path_data = os.path.join(os.getcwd(), 'data', 'wiki1m_for_simcse.txt')
    with open(path_data, 'r') as f:
        list_text = f.readlines()
    len(list_text)
    list_text = random.sample(list_text, num_sent)
    
    tokenized_inputs = tokenizer(list_text, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in tokenized_inputs['input_ids'].tolist()
    ]
    special_tokens_mask = (~torch.tensor(special_tokens_mask, dtype=torch.bool)).float()
    all_elements_are_zero = special_tokens_mask.sum(dim=-1) == 0
    special_tokens_mask = torch.where(all_elements_are_zero[:, None], torch.cat([torch.ones((special_tokens_mask.shape[0], 1)), special_tokens_mask[:, 1:]], dim=1), special_tokens_mask) # a subset of `special_tokens_mask` where all elements are 0 raises an error, so fill their 2nd index with 1.
    list_randint = torch.multinomial(special_tokens_mask, MAX_LEN - 2, replacement=False)
    
    list_token_type = ['mask', 'unk', 'pad']
    list_step = range(1, step_max + 1)
    # n_perturbate, step, i = 1, 1, 0
    dict_norm = {}
    df = pd.DataFrame(columns=list_token_type, index=pd.MultiIndex.from_product([range(1, n_perturbate_max + 1), list_step], names=['n_perturbate', 'step']))
    for step in tqdm(list_step, leave=False):
        num_dup = min(list_randint.size(1) // step + 1, n_perturbate_max + 1)
        list_n_perturbate = range(1, num_dup)
        tokenized_inputs_interleaved = {id_type: tensor.repeat_interleave(num_dup, dim=0) for id_type, tensor in tokenized_inputs.items()}
        
        for token_type in tqdm(list_token_type, leave=False):
            token_id = getattr(tokenizer, f'{token_type}_token_id')
            
            for n_perturbate in tqdm(list_n_perturbate, leave=False):
                for i in range(step):
                    start_idx = (n_perturbate - 1) * step + i
                    target_index_col = list_randint[:, start_idx:start_idx + 1]
                    target_index_col = target_index_col.repeat_interleave(num_dup - n_perturbate, dim=0).squeeze()
                    target_index_row = torch.Tensor([x for x in range(len(list_text) * num_dup) if x % num_dup not in range(n_perturbate)]).long()
                    target_index = target_index_row, target_index_col
                    special_tokens_mask_temp = special_tokens_mask.repeat_interleave(num_dup, dim=0)
                    value_to_replace_with = (torch.Tensor([token_id] * len(target_index_row)) * special_tokens_mask_temp[target_index]).long()
                    tokenized_inputs_interleaved['input_ids'][target_index] = value_to_replace_with
                    # print(tokenized_inputs_interleaved['input_ids'][:num_dup + 1])
            
            with torch.no_grad():
                _ = model.eval()
                list_corr, tsr_norm = calculate_corr(tokenized_inputs_interleaved, num_dup)
            df.loc[pd.MultiIndex.from_product([range(1, num_dup), [step]]), token_type] = list_corr
        dict_norm[step] = tsr_norm
    return df.astype(float), dict_norm

DEVICE='cuda'
BATCH_SIZE=4096
MAX_LEN=32

list_model_name = [
    # "princeton-nlp/unsup-simcse-bert-base-uncased",
    # "princeton-nlp/unsup-simcse-bert-large-uncased",
    # "princeton-nlp/unsup-simcse-roberta-base",
    "princeton-nlp/unsup-simcse-roberta-large"
]

# model_name = "princeton-nlp/unsup-simcse-bert-base-uncased"
for model_name in list_model_name:
    model = SimCSE(model_name)

    tokenizer = model.tokenizer
    model = model.model
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        BATCH_SIZE *= torch.cuda.device_count()
    _ = model.to(DEVICE)

    df, dict_norm = experiment(num_sent=100000, n_perturbate_max=9, step_max=9)
    df = df.dropna(axis=0, how='all')
    df['mean'] = df.mean(axis=1)
    df = df.round(4)
    x = model_name.split('/')[1].split('-')
    df.to_csv(f"../sdnorm_token_{x[2]}-{x[3]}.csv")
    with open(f'../sdnorm_token_{x[2]}-{x[3]}.pickle', 'wb') as f:
        pickle.dump(dict_norm, f)


# # similarity experiment
# model.similarity('I don\'t like this movie', 'I like this movie')
# model.similarity('I don\'t like this movie', 'I hate this movie')

# cosine_similarity(
#     (model.encode('He has a dog', normalize_to_unit=False) - model.encode('She has a dog', normalize_to_unit=False)).reshape(1, -1),
#     (model.encode('He plays soccer', normalize_to_unit=False) - model.encode('She plays soccer', normalize_to_unit=False)).reshape(1, -1)
# ) # 0.6591

# ((model.encode('He sleeps at night', normalize_to_unit=False) - model.encode('She sleeps at night', normalize_to_unit=False)) - (model.encode('He goes to school', normalize_to_unit=False) - model.encode('She goes to schools', normalize_to_unit=False))).norm() # 6.1194

# (model.encode('zzzzz', normalize_to_unit=False) - model.encode('z', normalize_to_unit=False))
# (model.encode('rrrrr', normalize_to_unit=False) - model.encode('r', normalize_to_unit=False))

# sentences_a = ['😍 😍 😍 😍']
# sentences_b = ['😍 😍 😍']
# model.similarity(sentences_a, sentences_b)