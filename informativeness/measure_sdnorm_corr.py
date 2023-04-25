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

def get_norm(tokenized_inputs_batch):
    global DEVICE
    tokenized_inputs_batch = {k: v.to(DEVICE) for k, v in tokenized_inputs_batch.items()}
    outputs = model(**tokenized_inputs_batch, return_dict=True)
    embeddings = outputs.last_hidden_state[:, 0]
    return embeddings.norm(dim=-1).cpu().detach()#.numpy().round(4)

def calculate_corr(tokenized_inputs_interleaved, n_perturbate=1, step=1):
    global BATCH_SIZE
    num_sent_in_batch = BATCH_SIZE // (n_perturbate + 1)
    adjusted_batch_size = num_sent_in_batch * (n_perturbate + 1)
    
    idx_to_select = [x for x in range(tokenized_inputs_interleaved['input_ids'].shape[0]) if x % 10 in range(n_perturbate + 1)]
    tokenized_inputs_selected = {id_type: tensor[idx_to_select] for id_type, tensor in tokenized_inputs_interleaved.items()}

    total_len = tokenized_inputs_selected['input_ids'].shape[0]
    list_corr = []
    for i in tqdm(range(0, total_len, adjusted_batch_size), leave=False):
        tokenized_inputs_batch = {id_type: tensor[i:i + adjusted_batch_size] for id_type, tensor in tokenized_inputs_selected.items()}
        
        # tokenized_inputs[token_type] = tokenized_inputs[token_type].reshape(num_sent_in_batch * (n_perturbate + 1), -1)
        result_att = get_norm(tokenized_inputs_batch)
        result_att = result_att.reshape(-1, n_perturbate + 1)
        x = torch.argsort(result_att).float()
        y = torch.Tensor([range(n_perturbate, -1, -1) for _ in range(result_att.shape[0])])
        spearman_corr = 1 - (x - y).pow(2).sum(dim=1).mul(6).div((n_perturbate + 1) * ((n_perturbate + 1) ** 2 - 1))
        list_corr.append(spearman_corr)
    corr_mean = torch.cat(list_corr).mean()
    return corr_mean.item()

def experiment(n_perturbate_max=9, step_max=9):
    path_data = os.path.join(os.getcwd(), 'data', 'wiki1m_for_simcse.txt')
    with open(path_data, 'r') as f:
        list_text = f.readlines()
    len(list_text)
    
    tokenized_inputs = tokenizer(list_text, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in tokenized_inputs['input_ids'].tolist()
    ]
    special_tokens_mask = (~torch.tensor(special_tokens_mask, dtype=torch.bool)).float()
    all_elements_are_zero = special_tokens_mask.sum(dim=-1) == 0
    special_tokens_mask = torch.where(all_elements_are_zero[:, None], torch.cat([torch.ones((special_tokens_mask.shape[0], 1)), special_tokens_mask[:, 1:]], dim=1), special_tokens_mask) # a subset of `special_tokens_mask` where all elements are 0 raises an error, so fill their 2nd index with 1.
    list_randint = torch.multinomial(special_tokens_mask, MAX_LEN - 2, replacement=False)
    
    dict_token_type = {
        'mask': '[MASK]',
        'unk': '[UNK]',
        'pad': '[PAD]'
    }
    
    list_n_perturbate = range(1, n_perturbate_max + 1)
    list_step = range(1, step_max + 1)
    df = pd.DataFrame(columns=['unk', 'mask', 'pad'], index=pd.MultiIndex.from_product([list_n_perturbate, list_step], names=['n_perturbate', 'step']))
    # n_perturbate, step = 1, 2
    for step in tqdm(list_step, leave=False):
        tokenized_inputs_interleaved = {id_type: tensor.repeat_interleave(n_perturbate_max + 1, dim=0) for id_type, tensor in tokenized_inputs.items()}
        for n_perturbate in tqdm(list_n_perturbate, leave=False):
            if n_perturbate * step > list_randint.size(1):
                continue
            for token_type, token_name in dict_token_type.items():
                token_id = tokenizer.convert_tokens_to_ids(token_name)
                for i in range(step):
                    start_idx = (n_perturbate - 1) * step + i
                    target_index_col = list_randint[:, start_idx:start_idx + 1]
                    target_index_col = target_index_col.repeat_interleave(n_perturbate_max - n_perturbate + 1, dim=0).squeeze()
                    target_index_row = torch.Tensor([x for x in range(len(list_text) * (n_perturbate_max + 1)) if x % 10 not in range(n_perturbate)]).long()
                    target_index = target_index_row, target_index_col
                    special_tokens_mask_temp = special_tokens_mask.repeat_interleave(n_perturbate_max + 1, dim=0)
                    value_to_replace_with = (torch.Tensor([token_id] * len(target_index_row)) * special_tokens_mask_temp[target_index]).long()
                    tokenized_inputs_interleaved['input_ids'][target_index] = value_to_replace_with
                    # print(tokenized_inputs_interleaved['input_ids'][10:20])

                    with torch.no_grad():
                        model.eval()
                        result = calculate_corr(tokenized_inputs_interleaved, n_perturbate=n_perturbate, step=step)
                    df.loc[(n_perturbate, step), token_type] = result
    return df.astype(float)

DEVICE='cuda'
BATCH_SIZE=512
MAX_LEN=32

list_model_name = [
    "princeton-nlp/unsup-simcse-bert-base-uncased",
    "princeton-nlp/unsup-simcse-bert-large-uncased",
    "princeton-nlp/unsup-simcse-roberta-base",
    "princeton-nlp/unsup-simcse-roberta-large"
]

for model_name in list_model_name:
    model = SimCSE(model_name)

    tokenizer = model.tokenizer
    model = model.model
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        BATCH_SIZE *= torch.cuda.device_count() * 4
    _ = model.to(DEVICE)

    df = experiment()
    df = df.dropna(axis=0, how='all')
    df['mean'] = df.mean(axis=1)
    df = df.round(4)
    x = model_name.split('/')[1].split('-')
    df.to_csv(f"../sdnorm_token_{x[2]}-{x[3]}.csv")


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