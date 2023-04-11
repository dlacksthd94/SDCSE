from simcse import SimCSE
from diffcse import DiffCSE
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import pandas as pd

# model = DiffCSE("voidism/diffcse-bert-base-uncased-sts")
model = SimCSE("princeton-nlp/unsup-simcse-bert-base-uncased")
# model = SimCSE("princeton-nlp/unsup-simcse-roberta-base")
# model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
# model = SimCSE("princeton-nlp/sup-simcse-roberta-base")
print(model.pooler)

def get_token_and_norm(tokenized_inputs):
    # [print(model.tokenizer.convert_ids_to_tokens(tokenized_inputs_)) for tokenized_inputs_ in tokenized_inputs['input_ids'].tolist()]
    tokenized_inputs = {k: v.to('cuda') for k, v in tokenized_inputs.items()}
    _ = model.model.cuda()
    outputs = model.model(**tokenized_inputs, return_dict=True)
    embeddings = outputs.last_hidden_state[:, 0]
    return embeddings.norm(dim=-1).cpu().detach().numpy().round(4)
    # print(model.encode(text, device='cuda', normalize_to_unit=False).norm(dim=-1))

def perturbate_inputs(text, n_perturbate=2, n_sim=100, step=2, perturbate_type=['att', 'unk', 'mask', 'pad']):
    # text = 'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as noise.'
    list_result = []

    # prepare
    list_text = [text] * (n_perturbate + 1)
    tokenized_inputs = model.tokenizer(list_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    len_token = tokenized_inputs['input_ids'][0].shape[0]
    assert len_token - 2 >= n_perturbate * step

    # use attention mask
    if 'att' in perturbate_type:
        list_corr_att = []
        for _ in tqdm(range(n_sim), leave=False):
            tokenized_inputs = model.tokenizer(list_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
            list_randint = np.random.choice(range(1, len_token - 1), n_perturbate * step, replace=False)
            for i in range(n_perturbate + 1):
                tokenized_inputs['attention_mask'][i][list_randint[:i * step]] = torch.Tensor([0] * i * step).long()
            result_att = get_token_and_norm(tokenized_inputs)
            list_corr_att.append(spearmanr(result_att, sorted(range(n_perturbate + 1), reverse=True))[0])
        result = round(sum(list_corr_att) / len(list_corr_att), 4)
        list_result.append(result)

    # use unk token
    if 'unk' in perturbate_type:
        list_corr_unk = []
        for _ in tqdm(range(n_sim), leave=False):
            tokenized_inputs = model.tokenizer(list_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
            list_randint = np.random.choice(range(1, len_token - 1), n_perturbate * step, replace=False)
            for i in range(n_perturbate + 1):
                tokenized_inputs['input_ids'][i][list_randint[:i * step]] = torch.Tensor([model.tokenizer.unk_token_id] * i * step).long()
            result_unk = get_token_and_norm(tokenized_inputs)
            list_corr_unk.append(spearmanr(result_unk, sorted(range(n_perturbate + 1), reverse=True))[0])
        result = round(sum(list_corr_unk) / len(list_corr_unk), 4)
        list_result.append(result)

    # use mask token
    if 'mask' in perturbate_type:
        list_corr_mask = []
        for _ in tqdm(range(n_sim), leave=False):
            tokenized_inputs = model.tokenizer(list_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
            list_randint = np.random.choice(range(1, len_token - 1), n_perturbate * step, replace=False)
            for i in range(n_perturbate + 1):
                tokenized_inputs['input_ids'][i][list_randint[:i * step]] = torch.Tensor([model.tokenizer.mask_token_id] * i * step).long()
            result_mask = get_token_and_norm(tokenized_inputs)
            list_corr_mask.append(spearmanr(result_mask, sorted(range(n_perturbate + 1), reverse=True))[0])
        result = round(sum(list_corr_mask) / len(list_corr_mask), 4)
        list_result.append(result)

    # use pad token
    if 'pad' in perturbate_type:
        list_corr_pad = []
        for _ in tqdm(range(n_sim), leave=False):
            tokenized_inputs = model.tokenizer(list_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
            list_randint = np.random.choice(range(1, len_token - 1), n_perturbate * step, replace=False)
            for i in range(n_perturbate + 1):
                tokenized_inputs['input_ids'][i][list_randint[:i * step]] = torch.Tensor([model.tokenizer.pad_token_id] * i * step).long()
            result_pad = get_token_and_norm(tokenized_inputs)
            list_corr_pad.append(spearmanr(result_pad, sorted(range(n_perturbate + 1), reverse=True))[0])
        result = round(sum(list_corr_pad) / len(list_corr_pad), 4)
        list_result.append(result)
        
    return list_result

def experiment(text, n_sim, perturbate_type):
    # original
    result_original = model.encode(text, normalize_to_unit=False).norm(dim=-1)
    norm = round(float(result_original), 4)
    print(norm)
    
    list_n_perturbate = [1, 2, 4]
    list_step = [1, 2, 4]
    df = pd.DataFrame(columns=perturbate_type, index=pd.MultiIndex.from_product([list_n_perturbate, list_step], names=['n_perturbate', 'step']))
    for n_perturbate in tqdm(list_n_perturbate, leave=False):
        for step in tqdm(list_step, leave=False):
            try:
                result = perturbate_inputs(text, n_perturbate=n_perturbate, n_sim=n_sim, step=step, perturbate_type=perturbate_type)
                df.loc[(n_perturbate, step)] = result
            except:
                pass
    return df.astype(float)

text = 'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as noise.' # 14.1141
df1 = experiment(text, n_sim=1000, perturbate_type=['att', 'unk', 'mask', 'pad'])
df1.loc[('argsort', 'argsort'), :] = df1.apply(lambda row: row.argsort(), axis=1).sum()
df1.loc[('mean', 'all'), :] = df1[:-1].mean()
df1
df1.astype(float).groupby('n_perturbate').mean()[:-2]
df1.astype(float).groupby('step').mean()[:-2]

text = 'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework.' # 13.901
df2 = experiment(text, n_sim=1000, perturbate_type=['att', 'unk', 'mask', 'pad'])
df2.loc[('argsort', 'argsort'), :] = df2.apply(lambda row: row.argsort(), axis=1).sum()
df2.loc[('mean', 'all'), :] = df2[:-1].mean()
df2
df2.astype(float).groupby('n_perturbate').mean()[:-2]
df2.astype(float).groupby('step').mean()[:-2]

text = 'Unsupervised SimCSE simply takes an input sentence' # 13.7006
df3 = experiment(text, n_sim=1000, perturbate_type=['att', 'unk', 'mask', 'pad'])
df3.loc[('argsort', 'argsort'), :] = df3.apply(lambda row: row.argsort(), axis=1).sum()
df3.loc[('mean', 'all'), :] = df3[:-1].mean()
df3
df3.astype(float).groupby('n_perturbate').mean()[:-2]
df3.astype(float).groupby('step').mean()[:-2]

# similarity experiment
model.similarity('I don\'t like this movie', 'I like this movie')
model.similarity('I don\'t like this movie', 'I hate this movie')

cosine_similarity(
    (model.encode('He has a dog', normalize_to_unit=False) - model.encode('She has a dog', normalize_to_unit=False)).reshape(1, -1),
    (model.encode('He plays soccer', normalize_to_unit=False) - model.encode('She plays soccer', normalize_to_unit=False)).reshape(1, -1)
) # 0.6591

((model.encode('He sleeps at night', normalize_to_unit=False) - model.encode('She sleeps at night', normalize_to_unit=False)) - (model.encode('He goes to school', normalize_to_unit=False) - model.encode('She goes to schools', normalize_to_unit=False))).norm() # 6.1194

(model.encode('zzzzz', normalize_to_unit=False) - model.encode('z', normalize_to_unit=False))
(model.encode('rrrrr', normalize_to_unit=False) - model.encode('r', normalize_to_unit=False))

sentences_a = ['😍 😍 😍 😍']
sentences_b = ['😍 😍 😍']
model.similarity(sentences_a, sentences_b)


########## informativeness
import pandas as pd
df = pd.read_csv(f'result/result_subsentence_informativeness_{"lg"}{"_large"}{""}_{"orig"}.csv', header=[0,1,2,3], index_col=0)
df.loc[:, pd.IndexSlice['unsup', :, 'bert', 'wo_p']]

########## embedding test
import pickle
import numpy as np
with open(f'data/{"wiki1m"}_{"mcse"}_{"unsup"}_{"bert"}_embedding_subsentence_{"lg"}{"_large"}{""}.pickle', 'rb') as f:
    list_embeddings_1 = pickle.load(f)
    
with open(f'data/{"wiki1m"}_{"diffcse"}_{"unsup"}_{"bert"}_embedding_subsentence_{"lg"}{"_large"}{""}.pickle', 'rb') as f:
    list_embeddings_2 = pickle.load(f)
    
for i, (embedding_1, embedding_2) in enumerate(zip(list_embeddings_1[3:], list_embeddings_2[3:])):
    assert (embedding_1[1] == embedding_2[1]).sum() == 0 # assert that any value is not same

(embedding_1[1] - embedding_2[1]).mean()
np.linalg.norm(embedding_1[1])
np.linalg.norm(embedding_2[1])

from tqdm import tqdm
import numpy as np
temp_sum = np.random.randn(15, 15)
for i in tqdm(range(10**6)):
    temp = np.random.randn(15, 15)
    temp_sum *= temp