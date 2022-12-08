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
parser.add_argument('-e', required=False, help='select sentence encoder [bert, sbert, simcse, diffcse, promcse]', type=str, choices=['bert', 'sbert', 'simcse', 'diffcse', 'promcse'], default='simcse')
parser.add_argument('-p', required=False, help='select constituency parser from [base, large]', type=str, choices=['base', 'large'], default='base')
parser.add_argument('-pp', required=False, help='select constituency parser pipeline [sm, md, lg]', type=str, choices=['sm', 'md', 'lg'], default='lg')
parser.add_argument('-d', required=False, help='select dataset from [wiki1m, STS12, STS13, STS14, STS15, STS16, STS-B, SICK-R]', type=str, choices=['wiki1m', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R'], default='wiki1m')

DATASET = parser.parse_args().d
BATCH_SIZE = 64 # 32 or 64
MODE = parser.parse_args().m
GPU_ID = parser.parse_args().g
ENCODER = parser.parse_args().e
PARSER = 'benepar_en3' if parser.parse_args().p == 'base' else 'benepar_en3_large'
PIPELINE = f'en_core_web_{parser.parse_args().pp}'
N = ''
print(DATASET, MODE, ENCODER)

# load tokenizer & model
if ENCODER == 'bert':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
elif ENCODER == 'sbert':
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/nli-bert-base")
    model = AutoModel.from_pretrained("sentence-transformers/nli-bert-base")
elif ENCODER == 'simcse':
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
elif ENCODER == 'diffcse':
    tokenizer = AutoTokenizer.from_pretrained("voidism/diffcse-bert-base-uncased-sts")
    model = AutoModel.from_pretrained("voidism/diffcse-bert-base-uncased-sts")
# elif ENCODER == 'promcse':
#     tokenizer = AutoTokenizer.from_pretrained()
#     model = AutoModel.from_pretrained("unsup-promcse-bert-base.bin")

if MODE == 'base':
    if not os.path.exists(f'data/{DATASET}_{ENCODER}_tokenized.pickle'):
        # load dataset
        with open(f'../SimCSE/{DATASET + "_for_simcse" if DATASET == "wiki1m" else DATASET}.txt') as f:
            list_text = f.readlines()
                        
        # preprocessing
        for i in tqdm(range(len(list_text))):
            list_text[i] = list_text[i].strip()

        # make batch to load on GPU
        list_batch = []
        for i in tqdm(range(0, len(list_text), BATCH_SIZE)):
            batch = tokenizer(text=list_text[i:i+BATCH_SIZE], padding=True, truncation=True, return_tensors="pt", verbose=True, max_length=510)
            list_batch.append(batch)

        with open(f'data/{DATASET}_{ENCODER}_tokenized.pickle', 'wb') as f:
            pickle.dump(list_batch, f)
    else:
        with open(f'data/{DATASET}_{ENCODER}_tokenized.pickle', 'rb') as f:
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

    with open(f'data/{DATASET}_{ENCODER}_embedding.pickle', 'wb') as f:
        pickle.dump(embeddings.tolist(), f)
        
elif MODE == 'sub':
    if not os.path.exists(f'data/{DATASET}_{ENCODER}_tokenized_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle'):
        # load dataset
        with open(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}_subsentence.pickle', 'rb') as f:
            list_subsentence = pickle.load(f)

        # make batch to load on GPU
        list_batch = []
        for subsentence in tqdm(list_subsentence):
            batch = tokenizer(text=subsentence, padding=True, truncation=True, return_tensors="pt", verbose=True)
            list_batch.append(batch)

        with open(f'data/{DATASET}_{ENCODER}_tokenized_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'wb') as f:
            pickle.dump(list_batch, f)
    else:
        with open(f'data/{DATASET}_{ENCODER}_tokenized_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'rb') as f:
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

    with open(f'data/{DATASET}_{ENCODER}_embedding_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'wb') as f:
        pickle.dump(embeddings, f)