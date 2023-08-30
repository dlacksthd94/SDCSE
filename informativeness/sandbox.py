import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, AutoModel
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import os

# BERT 모델 불러오기
model_name = 'princeton-nlp/unsup-simcse-bert-base-uncased'
# model_name = 'princeton-nlp/unsup-simcse-bert-large-uncased'
# model_name = 'princeton-nlp/unsup-simcse-roberta-base'
# model_name = 'princeton-nlp/unsup-simcse-roberta-large'
# model_name = 'bert-base-uncased'
# model_name = os.path.join(os.getcwd(), 'result', 'mask0/')
# model_name = os.path.join(os.getcwd(), 'result', 'mask0_cls0/')
# model_name = os.path.join(os.getcwd(), 'result', 'mask0_cls0_sep0/')
# model_name = os.path.join(os.getcwd(), 'result', 'mask0_sep0/')
# model_name = 'bert-large-uncased'

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.embeddings.word_embeddings.weight[100].norm() # unk 100, cls 101, sep 102, mask 103, pad 0
model.embeddings.word_embeddings.weight[101].norm() # unk 100, cls 101, sep 102, mask 103, pad 0
model.embeddings.word_embeddings.weight[102].norm() # unk 100, cls 101, sep 102, mask 103, pad 0
model.embeddings.word_embeddings.weight[103].norm() # unk 100, cls 101, sep 102, mask 103, pad 0
(model.embeddings.word_embeddings.weight[100] / 100).norm()

list_sentence = [
    'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as noise.',
    'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework.',
    'Unsupervised SimCSE simply takes an input sentence',
    '----------',
    'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as noise.',
    'Unsupervised [MASK] simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as [MASK].',
    'Unsupervised [MASK] simply takes an input sentence and [MASK] itself in a contrastive learning framework, with only [MASK] dropout used as [MASK].',
    '----------',
    'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as noise.',
    'Unsupervised [UNK] simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as [UNK].',
    'Unsupervised [UNK] simply takes an input sentence and [UNK] itself in a contrastive learning framework, with only [UNK] dropout used as [UNK].',
    '----------',
    'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as noise.',
    'Unsupervised [PAD] simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as [PAD].',
    'Unsupervised [PAD] simply takes an input sentence and [PAD] itself in a contrastive learning framework, with only [PAD] dropout used as [PAD].',
    '----------',
    '----------',
    'Chelsea have signed one of the biggest attacking talents on the planet.',
    'Chelsea have signed one of the biggest attacking talents.',
    'Chelsea have signed attacking talents.',
    '----------',
    'Chelsea have signed one of the biggest attacking talents on the planet.',
    'Chelsea have signed [MASK] of the biggest attacking talents on the planet.',
    'Chelsea have signed [MASK] of the biggest [MASK] talents on the planet.',
    '----------',
    'Chelsea have signed one of the biggest attacking talents on the planet.',
    'Chelsea have signed [UNK] of the biggest attacking talents on the planet.',
    'Chelsea have signed [UNK] of the biggest [UNK] talents on the planet.',
    '----------',
    'Chelsea have signed one of the biggest attacking talents on the planet.',
    'Chelsea have signed [PAD] of the biggest attacking talents on the planet.',
    'Chelsea have signed [PAD] of the biggest [PAD] talents on the planet.',
    '----------',
    '----------',
    'x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x',
    'x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x',
    'x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x',
    'x x x x x x x x x x x x x x x x x x x x',
    'x x x x x x x x x x',
    '----------',
    '----------',
    '[MASK]',
    '[UNK]',
    '[PAD]',
    '<mask>',
    '<unk>',
    '<pad>',
]

list_sentence = [
    'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as noise.',
    'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework.',
    'Unsupervised SimCSE takes an input sentence and predicts itself',
    '----------',
    'Chelsea have signed one of the biggest attacking talents on the planet.',
    'Chelsea have signed one of the biggest attacking talents.',
    'Chelsea have signed an attacking talent.',
    '----------',
    'Walking through the tranquil garden brought me a sense of peace and serenity.',
    'Walking through the garden brought me peace and serenity.',
    'Walking brought me peace and serenity.',
    '----------',
    'Rapid advancements in technology are shaping the way we communicate, work, and interact with the world around us, revolutionizing various aspects of our daily lives.',
    'Advancements in technology are shaping the way we communicate, work, and interact with the world, revolutionizing our daily lives.',
    'Advancements in technology are shaping the way we communicate, work, and interact with the world.',
    '----------',
    'The old cottage nestled in the valley was rumored to be haunted, adding an air of mystery to the surrounding woods.',
    'The old cottage was rumored to be haunted, adding an air of mystery to the surrounding woods.',
    'The old cottage was rumored to be haunted, adding an air of mystery',
]

sent = '[MASK]'
for sent in list_sentence:
    if sent != '----------':
        input = tokenizer(sent, return_tensors='pt')
        output = model(**input, return_dict=True)
        # cos_sim = cosine_similarity(output.last_hidden_state.squeeze().detach().numpy()).round(2)
        # cos_sim.mean()
        norm_wop = output.last_hidden_state[:, 0, :].squeeze().norm().item()
        norm_wp = output.pooler_output.squeeze().norm().item()
        round(norm_wop, 2), round(norm_wp, 2)
    else:
        '----------'

A = model.encoder.layer[11].attention.self.query.weight#[:96, :96]
B = model.encoder.layer[11].attention.self.key.weight#[:96, :96]
AB = torch.matmul(A.T, B)
eigenvalues, eigenvectors = torch.eig(AB, eigenvectors=True)
(eigenvalues[:, 0] >= 0).sum() / eigenvalues.shape[0]


A = model.encoder.layer[11].attention.self.query.weight[:96, :96]
A.T * A
Q, R = torch.qr(A)
I = torch.eye(Q.shape[-1])
result = torch.allclose(torch.matmul(Q.transpose(-2, -1), Q), I, rtol=1e-05, atol=1e-08)
torch.allclose(torch.matmul(Q.transpose(-2, -1), Q), I, rtol=1e-05, atol=1e-06)

A = model.encoder.layer[11].attention.self.key.weight[:96, :96]
A.T * A
Q, R = torch.qr(A)
I = torch.eye(Q.shape[-1])
result = torch.allclose(torch.matmul(Q.transpose(-2, -1), Q), I, rtol=1e-05, atol=1e-08)
torch.allclose(torch.matmul(Q.transpose(-2, -1), Q), I, rtol=1e-05, atol=1e-06)

A = model.encoder.layer[11].attention.self.value.weight[:96, :96]
A.T * A
Q, R = torch.qr(A)
I = torch.eye(Q.shape[-1])
result = torch.allclose(torch.matmul(Q.transpose(-2, -1), Q), I, rtol=1e-05, atol=1e-08)
torch.allclose(torch.matmul(Q.transpose(-2, -1), Q), I, rtol=1e-05, atol=1e-06)


PATH_DATA = os.path.join(os.getcwd(), 'data', 'wiki1m_for_simcse.txt')
with open(PATH_DATA, 'r') as f:
    list_text = f.readlines()
len(list_text)

arr_seq_len = np.array([len(x) for x in tokenizer(list_text)['input_ids']])
arr_seq_len.mean()
sum(arr_seq_len < 32)
sum(arr_seq_len < 64)
sum(arr_seq_len < 128)
# tokenizer(list_text, return_tensors='pt', padding=True, truncation=True, max_length=32)

""" """ """ """ """ cov exp """ """ """ """ """
import numpy as np
from itertools import permutations
# l = np.array(list(permutations([1,2,3], 2)))
# l = np.array([[1,3], [1,1], [3,3], [3,1]])
pool = range(1, 4)
l = np.array([[i, j] for i in pool for j in pool])
l[0] = [0,0]
l[-1] = [3,3]
np.cov(l.T)


""" """ """ """ """ calculating hypersphere volume """ """ """ """ """
import math
from scipy.special import gamma

def hypersphere_volume(r, n):
    volume = math.pi**(n / 2) / (gamma(n / 2 + 1)) * r**n
    return volume

r = 9.9  # 반지름
n = 10  # 차원

hypersphere_volume(r, n)

""" """ """ """ """ calculating hypersphere surface area """ """ """ """ """
import math

def hypersphere_surface_area(r, n):
    surface_area = (2 * math.pi**(n / 2)) / gamma(n / 2) * r**(n - 1)
    return surface_area

r = 4.5  # 반지름
n = 400  # 차원

hypersphere_surface_area(r, n)

""" """ """ """ """ exploring hypersphere area and volumne """ """ """ """ """
def volume_area(r, N):
    print(*['n', 'v_n', 's_n-1', 'v_n+1', 's_n', 'v_n+2', 's_n+1'], sep='\t')
    for n in range(N):
        v_n = hypersphere_volume(r, n)
        v_n1 = hypersphere_volume(r, n + 1)
        v_n2 = v_n * 2 * math.pi / (n + 2) * r**2
        s_n_1 = hypersphere_surface_area(r, n)
        s_n = hypersphere_surface_area(r, n + 1)
        s_n1 = s_n_1 * 2 * math.pi / n * r**2
        # s_n = 2 * math.pi * v_n_1
        result = np.array([n, v_n, s_n_1, v_n1, s_n, v_n2, s_n1]).round(2)
        print(*result, sep='\t')

def volume_area_recurrent(r, N):
    print(*['n', 'v_n', 's_n', 'v_n_r', 's_n_r'], sep='\t')
    v1 = hypersphere_volume(r, 1)
    v2 = hypersphere_volume(r, 2)
    s0 = hypersphere_surface_area(r, 1)
    s1 = hypersphere_surface_area(r, 2)
    list_v = [v1, v2]
    list_s = [s0, s1]
    print(1, round(v1, 2), round(s0, 2), sep='\t')
    print(2, round(v2, 2), round(s1, 2), sep='\t')
    for n in range(3, N):
        # v_n = hypersphere_volume(r, n)
        # s_n_1 = hypersphere_surface_area(r, n)
        v_n = 0
        s_n_1 = 0
        v_n_r = list_v[n - 3] * 2 * math.pi / (n) * r**2
        s_n_1_r = list_s[n - 3] * 2 * math.pi / (n - 2) * r**2
        list_v.append(v_n_r)
        list_s.append(s_n_1_r)
        result = np.array([n, v_n, s_n_1, list_v[n - 1], list_s[n - 1]]).round(2)
        print(*result, sep='\t')
    return list_v, list_s

r, N=14.4, 768
# volume_area(r, N)
list_v, list_s = volume_area_recurrent(r, N)
volume_area_recurrent(14.5, 768)[1][-1] / volume_area_recurrent(14.4, 768)[1][-1]
(14.5 / 14.4)**767
volume_area_recurrent(14.5, 768)[0][-1] / volume_area_recurrent(14.4, 768)[0][-1]
(volume_area_recurrent(14.5, 768)[0][-1] - volume_area_recurrent(14.4, 768)[0][-1]) / (volume_area_recurrent(14.4, 768)[0][-1] - volume_area_recurrent(14.3, 768)[0][-1])

volume_area_recurrent(2, 10)[1][-1] / volume_area_recurrent(1.5, 10)[1][-1]
volume_area_recurrent(2, 10)[0][-1] / volume_area_recurrent(1.5, 10)[0][-1]