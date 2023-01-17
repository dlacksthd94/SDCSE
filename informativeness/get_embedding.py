import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import pickle
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# arg
parser = argparse.ArgumentParser()
parser.add_argument('-m', required=False, help='select mode from [base, sub, mask]', type=str, choices=['base', 'sub', 'mask'], default='sub')
parser.add_argument('-g', required=False, help='select which gpu to run on', type=int, choices=[0, 1, 2, 3], default=0)
parser.add_argument('-e', required=False, help='select sentence encoder [bert, sbert, simcse, diffcse, promcse]', type=str, choices=['bert', 'sbert', 'simcse', 'diffcse', 'promcse'], default='promcse')
parser.add_argument('-p', required=False, help='select constituency parser from [base, large]', type=str, choices=['base', 'large'], default='base')
parser.add_argument('-pp', required=False, help='select constituency parser pipeline [sm, md, lg]', type=str, choices=['sm', 'md', 'lg'], default='lg')
parser.add_argument('-d', required=False, help='select dataset from [wiki1m, STS12, STS13, STS14, STS15, STS16, STS-B, SICK-R, nli, quora, simplewiki, specter, covid]', type=str, choices=['wiki1m', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R', 'nli', 'quora', 'simplewiki', 'specter', 'covid', 'huffpost'], default='wiki1m')

DATASET = parser.parse_args().d
BATCH_SIZE = 64 # 32 or 64
MODE = parser.parse_args().m
GPU_ID = parser.parse_args().g
ENCODER = parser.parse_args().e
PARSER = 'benepar_en3' if parser.parse_args().p == 'base' else 'benepar_en3_large'
PIPELINE = f'en_core_web_{parser.parse_args().pp}'
N = ''

dict_dataset = {
    'bert': {'bert': "bert-base-uncased", 'roberta': "roberta-base"},
    'sbert': {'bert': "sentence-transformers/nli-bert-base", 'roberta': "sentence-transformers/nli-roberta-base"},
    'simcse': {'bert': "princeton-nlp/unsup-simcse-bert-base-uncased", 'roberta': "princeton-nlp/unsup-simcse-roberta-base"},
    'diffcse': {'bert': "voidism/diffcse-bert-base-uncased-sts", 'roberta': "voidism/diffcse-roberta-base-sts"},
    'promcse': {'bert': "YuxinJiang/unsup-promcse-bert-base-uncased", 'roberta': "YuxinJiang/sup-promcse-roberta-large"}
}

# load tokenizer & model
dict_tokenizer = {}
dict_model = {}
dict_plm = dict_dataset[ENCODER]
for plm in dict_plm:
    path = dict_plm[plm]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    dict_tokenizer[plm] = tokenizer
    dict_model[plm] = model

for plm in dict_plm:
    print(DATASET, MODE, ENCODER, plm)
    tokenizer = dict_tokenizer[plm]
    model = dict_model[plm]
    if MODE == 'base':
        if not os.path.exists(f'data/{DATASET}_{ENCODER}_{plm}_tokenized.pickle'):
            # load dataset
            with open(f'../SimCSE/{DATASET}.txt') as f:
                list_text = f.readlines()
                            
            # preprocessing
            for i in tqdm(range(len(list_text))):
                list_text[i] = list_text[i].strip()

            # make batch to load on GPU
            list_batch = []
            for i in tqdm(range(0, len(list_text), BATCH_SIZE)):
                batch = tokenizer(text=list_text[i:i+BATCH_SIZE], padding=True, truncation=True, return_tensors="pt", verbose=True, max_length=510)
                list_batch.append(batch)

            with open(f'data/{DATASET}_{ENCODER}_{plm}_tokenized.pickle', 'wb') as f:
                pickle.dump(list_batch, f)
        else:
            with open(f'data/{DATASET}_{ENCODER}_{plm}_tokenized.pickle', 'rb') as f:
                list_batch = pickle.load(f)
            
        for i in tqdm(range(len(list_batch))):
            _ = list_batch[i].to(f'cuda:{GPU_ID}')

        # get embeddings
        _ = model.to(f'cuda:{GPU_ID}')
        list_embeddings = []
        with torch.no_grad():
            for batch in tqdm(list_batch):
                embeddings_temp = model(**batch, output_hidden_states=True, return_dict=True).pooler_output
                list_embeddings.append(embeddings_temp)
        
        embeddings = torch.concat(list_embeddings).detach().cpu().numpy()

        with open(f'data/{DATASET}_{ENCODER}_{plm}_embedding.pickle', 'wb') as f:
            pickle.dump(embeddings.tolist(), f)
            
    elif MODE == 'sub':
        if not os.path.exists(f'data/{DATASET}_{ENCODER}_{plm}_tokenized_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle'):
            # load dataset
            with open(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}_subsentence.pickle', 'rb') as f:
                list_subsentence = pickle.load(f)

            # make batch to load on GPU
            list_batch = []
            for subsentence in tqdm(list_subsentence):
                batch = tokenizer(text=subsentence, padding=True, truncation=True, return_tensors="pt", verbose=True, max_length=510)
                list_batch.append(batch)

            with open(f'data/{DATASET}_{ENCODER}_{plm}_tokenized_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'wb') as f:
                pickle.dump(list_batch, f)
        else:
            with open(f'data/{DATASET}_{ENCODER}_{plm}_tokenized_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'rb') as f:
                list_batch = pickle.load(f)
        
        for i in tqdm(range(len(list_batch))):
            _ = list_batch[i].to(f'cuda:{GPU_ID}')

        # get embeddings
        _ = model.to(f'cuda:{GPU_ID}')
        list_embeddings = []
        with torch.no_grad():
            for batch in tqdm(list_batch):
                embeddings_temp = model(**batch, output_hidden_states=True, return_dict=True).pooler_output
                list_embeddings.append(embeddings_temp)

        embeddings = list(map(lambda embedding: embedding.detach().cpu().numpy(), list_embeddings))

        with open(f'data/{DATASET}_{ENCODER}_{plm}_embedding_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'wb') as f:
            pickle.dump(embeddings, f)

    elif MODE == 'mask':
        if not os.path.exists(f'data/{DATASET}_{ENCODER}_{plm}_tokenized_mask.pickle'):
            # load dataset
            with open(f'data/{DATASET}_masked.pickle', 'rb') as f:
                list_masked_sentence = pickle.load(f)

            # make batch to load on GPU
            list_batch = []
            for masked_sentence in tqdm(list_masked_sentence):
                batch = tokenizer(text=masked_sentence, padding=True, truncation=True, return_tensors="pt", verbose=True, max_length=510)
                list_batch.append(batch)

            with open(f'data/{DATASET}_{ENCODER}_{plm}_tokenized_mask.pickle', 'wb') as f:
                pickle.dump(list_batch, f)
        else:
            with open(f'data/{DATASET}_{ENCODER}_{plm}_tokenized_mask.pickle', 'rb') as f:
                list_batch = pickle.load(f)
        
        for i in tqdm(range(len(list_batch))):
            _ = list_batch[i].to(f'cuda:{GPU_ID}')

        # get embeddings
        _ = model.to(f'cuda:{GPU_ID}')
        list_embeddings = []
        with torch.no_grad():
            for batch in tqdm(list_batch):
                embeddings_temp = model(**batch, output_hidden_states=True, return_dict=True).pooler_output
                list_embeddings.append(embeddings_temp)

        embeddings = list(map(lambda embedding: embedding.detach().cpu().numpy(), list_embeddings))

        with open(f'data/{DATASET}_{ENCODER}_{plm}_embedding_mask.pickle', 'wb') as f:
            pickle.dump(embeddings, f)