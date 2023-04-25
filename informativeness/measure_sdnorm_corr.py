from asyncio.subprocess import DEVNULL
from multiprocessing import Barrier
from simcse import SimCSE
# from diffcse import DiffCSE
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import pandas as pd
import os

DEVICE='cuda'
BATCH_SIZE=32

# model = DiffCSE("voidism/diffcse-bert-base-uncased-sts")
model = SimCSE("princeton-nlp/unsup-simcse-bert-base-uncased")
# model = SimCSE("princeton-nlp/unsup-simcse-roberta-base")
# model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
# model = SimCSE("princeton-nlp/sup-simcse-roberta-base")
print(model.pooler)
print(model.device)

def get_norm(tokenized_inputs):
    global DEVICE
    # [print(model.tokenizer.convert_ids_to_tokens(tokenized_inputs_)) for tokenized_inputs_ in tokenized_inputs['input_ids'].tolist()]
    tokenized_inputs = {k: v.to(DEVICE) for k, v in tokenized_inputs.items()}
    # _ = model.model.to(DEVICE)
    outputs = model.model(**tokenized_inputs, return_dict=True)
    embeddings = outputs.last_hidden_state[:, 0]
    return embeddings.norm(dim=-1).cpu().detach()#.numpy().round(4)
    # print(model.encode(text, device=DEVICE, normalize_to_unit=False).norm(dim=-1))

def perturbate_inputs(text, n_perturbate=1, n_sim=10, step=1):
    # text = 'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as noise.'
    global BATCH_SIZE
    batch_size = BATCH_SIZE // (n_perturbate + 1)
    
    list_result = []

    # prepare
    tokenized_inputs = model.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    len_token = tokenized_inputs['input_ids'][0].shape[0]
    assert len_token - 2 >= n_perturbate * step
    list_text = [text] * (n_perturbate + 1) * batch_size

    dict_token_type = {
        'att': (0, 'attention_mask'),
        'unk': (model.tokenizer.unk_token_id, 'input_ids'),
        'mask': (model.tokenizer.mask_token_id, 'input_ids'),
        'pad': (model.tokenizer.pad_token_id, 'input_ids')
    }
    
    for _perturbate_type, (token_id, token_type) in dict_token_type.items():
        list_corr_temp = []
        for _ in tqdm(range(n_sim * n_perturbate // batch_size), leave=False):
            tokenized_inputs = model.tokenizer(list_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
            tokenized_inputs[token_type] = tokenized_inputs[token_type].reshape(batch_size, n_perturbate + 1, -1)
            list_randint = np.array([np.random.choice(range(1, len_token - 1), n_perturbate * step, replace=False) for _ in range(batch_size)])
            for i in range(1, n_perturbate + 1):
                target_row = tokenized_inputs[token_type][:, i, :]
                target_idx = list_randint[:, :step * i]
                for j in range(target_idx.shape[1]):
                    target_row[range(batch_size), target_idx[:, j]] = torch.Tensor([token_id] * batch_size).long()
            tokenized_inputs[token_type] = tokenized_inputs[token_type].reshape(batch_size * (n_perturbate + 1), -1)
            result_att = get_norm(tokenized_inputs)
            result_att = result_att.reshape(batch_size, n_perturbate + 1)
            x = torch.argsort(result_att).float()
            y = torch.Tensor([range(n_perturbate, -1, -1) for i in range(batch_size)])
            spearman_corr = 1 - (x - y).pow(2).sum(dim=1).mul(6).div((n_perturbate + 1) * ((n_perturbate + 1) ** 2 - 1))
            list_corr_temp.append(spearman_corr)
        result = round(float(torch.cat(list_corr_temp).mean()), 4)
        list_result.append(result)

    return list_result

def experiment(text, n_sim):
    # original
    result_original = model.encode(text, normalize_to_unit=False).norm(dim=-1)
    norm = round(float(result_original), 4)
    print(norm)
    
    list_n_perturbate = [1, 2, 3]
    list_step = [1, 2, 3]
    df = pd.DataFrame(columns=['att', 'unk', 'mask', 'pad'], index=pd.MultiIndex.from_product([list_n_perturbate, list_step], names=['n_perturbate', 'step']))
    for n_perturbate in tqdm(list_n_perturbate, leave=False):
        for step in tqdm(list_step, leave=False):
            try:
                result = perturbate_inputs(text, n_perturbate=n_perturbate, n_sim=n_sim, step=step)
                df.loc[(n_perturbate, step)] = result
            except:
                pass
    return df.astype(float)

PATH_DATA = os.path.join(os.getcwd(), 'data', 'wiki1m_for_simcse.txt')
with open(PATH_DATA, 'r') as f:
    list_text = f.readlines()
len(list_text)




text = list_text[3]
df1 = experiment(text, n_sim=100)
df1.loc[('mean', 'all'), :] = df1[:-1].mean()
df1
df1.astype(float).groupby('n_perturbate').mean()[:-1]
df1.astype(float).groupby('step').mean()[:-1]

text = 'The sound of laughter echoed through the room, filling me with a sense of joy and happiness.' # 13.901
df2 = experiment(text, n_sim=100)
df2.loc[('mean', 'all'), :] = df2[:-1].mean()
df2
df2.astype(float).groupby('n_perturbate').mean()[:-1]
df2.astype(float).groupby('step').mean()[:-1]

text = 'I love listening to music while I work.' # 13.7006
df3 = experiment(text, n_sim=100)
df3.loc[('mean', 'all'), :] = df3[:-1].mean()
df3
df3.astype(float).groupby('n_perturbate').mean()[:-1]
df3.astype(float).groupby('step').mean()[:-1]

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