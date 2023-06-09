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
model_name = 'bert-base-uncased'
model_name = os.path.join(os.getcwd(), 'result', 'mask0/')
model_name = os.path.join(os.getcwd(), 'result', 'mask0_cls0/')
model_name = os.path.join(os.getcwd(), 'result', 'mask0_cls0_sep0/')
model_name = os.path.join(os.getcwd(), 'result', 'mask0_sep0/')
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

sent = '[MASK]'
for sent in list_sentence:
    if sent != '----------':
        input = tokenizer(sent, return_tensors='pt')
        output = model(**input, return_dict=True)
        # cos_sim = cosine_similarity(output.last_hidden_state.squeeze().detach().numpy()).round(2)
        # cos_sim.mean()
        norm_wop = output.last_hidden_state[:, 0, :].squeeze().norm().item()
        norm_wp = output.pooler_output.squeeze().norm().item()
        norm_wop, norm_wp
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