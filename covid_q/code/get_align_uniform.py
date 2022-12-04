import torch
import numpy as np
import csv
from transformers import BertModel, BertTokenizer
# from methods import clean_line, save_to_pickle, read_csv, read_pickle # code 폴더에서 .py로 실행할 때는 이거 써야
from covid_q.code.methods import clean_line, save_to_pickle, read_csv, read_pickle # project 폴더의 주피터 파일에서 실행할 때는 이거 써야

# bsz : batch size (number of positive pairs)
# d   : latent dim
# x   : Tensor, shape=[bsz, d]
#       latents for one side of positive pairs
# y   : Tensor, shape=[bsz, d]
#       latents for the other side of positive pairs

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

# [BERT] Encode text
def get_embedding(input_string, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(input_string, add_special_tokens = True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0].numpy()  # Models outputs are now tuples
        # last_hidden_states = last_hidden_states[:, 0, :]
        last_hidden_states = np.mean(last_hidden_states, axis = 1)
        # last_hidden_states = np.amax(last_hidden_states, axis = 1)
        last_hidden_states = last_hidden_states.flatten()
        return torch.Tensor(last_hidden_states)


# [BERT] Gets all encodings given an input csv (for positive_pairs.csv)
def get_all_embeddings_align(input_csv, tokenizer, model):
    reader = read_csv(input_csv, True)

    embeddings_1 = list()
    embeddings_2 = list()

    for row in reader:
        question_1 = row[0]
        question_2 = row[1]
        embeddings_1.append(get_embedding(question_1, tokenizer, model))
        embeddings_2.append(get_embedding(question_2, tokenizer, model))

    emb1 = torch.stack(embeddings_1, 0)
    emb2 = torch.stack(embeddings_2, 0)
    # print('emb1 shape:', emb1.shape)
    # print('emb2 shape:', emb2.shape)

    return emb1, emb2


# [BERT] Gets all encodings given an input csv (for train4_uniform.csv)
def get_all_embeddings_uniform(input_csv, tokenizer, model):
    reader = read_csv(input_csv, True)

    embeddings = list()

    for row in reader:
        question = row[0]
        embeddings.append(get_embedding(question, tokenizer, model))

    emb = torch.stack(embeddings, 0)
    # print('emb shape:', emb.shape)

    return emb


# [BERT] Gets all encodings given an input csv (for final_master_dataset.csv)
def get_all_embeddings_uniform_all(input_csv, tokenizer, model):
    reader = read_csv(input_csv, True)

    embeddings = list()

    for row in reader:
        question = row[2]
        embeddings.append(get_embedding(question, tokenizer, model))

    emb = torch.stack(embeddings, 0)
    # print('emb shape:', emb.shape)

    return emb


# [SBERT, SimCSE] Get encodings for calculating alignment (for positive_pairs.csv)
def get_sentence_embedding_align(input_csv, model):
    reader = read_csv(input_csv, True)
    
    q1 = [row[0] for row in reader]
    q2 = [row[1] for row in reader]
    
    emb1 = torch.Tensor(model.encode(q1))
    emb2 = torch.Tensor(model.encode(q2))
    
    # print('emb1 shape:', emb1.shape)
    # print('emb2 shape:', emb2.shape)

    return emb1, emb2


# [SBERT, SimCSE] Get encodings for calculating uniformity (for train4_uniform.csv)
def get_sentence_embedding_uniform(input_csv, model):
    reader = read_csv(input_csv, True)
    
    q = [row[0] for row in reader]
    
    emb = torch.Tensor(model.encode(q))
    
    # print('emb shape:', emb.shape)

    return emb


# [SBERT, SimCSE] Get encodings for calculating uniformity (for final_master_dataset.csv)
def get_sentence_embedding_uniform_all(input_csv, model):
    reader = read_csv(input_csv, True)
    
    q = [row[2] for row in reader]
    
    emb = torch.Tensor(model.encode(q))
    
    # print('emb shape:', emb.shape)

    return emb
