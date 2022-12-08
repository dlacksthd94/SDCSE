from errno import ECOMM
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
parser.add_argument('-e', required=False, help='select sentence encoder [sbert, simcse, diffcse, promcse]', type=str, choices=['sbert', 'simcse', 'diffcse', 'promcse'], default='simcse')
parser.add_argument('-p', required=False, help='select constituency parser from [base, large]', type=str, choices=['base', 'large'], default='base')
parser.add_argument('-pp', required=False, help='select constituency parser pipeline [sm, md, lg]', type=str, choices=['sm', 'md', 'lg'], default='lg')

BATCH_SIZE = 64 # 32 or 64
MODE = parser.parse_args().m
GPU_ID = parser.parse_args().g

ENCODER = parser.parse_args().e
PARSER = 'benepar_en3' if parser.parse_args().p == 'base' else 'benepar_en3_large'
PIPELINE = f'en_core_web_{parser.parse_args().pp}'
N = '_' + str(parser.parse_args().n + 1) if parser.parse_args().n + 1 else ''

# load tokenizer & model
if ENCODER == 'sbert':
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
elif ENCODER == 'simcse':
    tokenizer = AutoTokenizer.from_pretrained("voidism/diffcse-bert-base-uncased-sts")
    model = AutoModel.from_pretrained("voidism/diffcse-bert-base-uncased-sts")
elif ENCODER == 'diffcse':
    tokenizer = AutoTokenizer.from_pretrained("voidism/diffcse-bert-base-uncased-sts")
    model = AutoModel.from_pretrained("voidism/diffcse-bert-base-uncased-sts")
elif ENCODER == 'promcse':
    tokenizer = AutoTokenizer.from_pretrained("voidism/diffcse-bert-base-uncased-sts")
    model = AutoModel.from_pretrained("voidism/diffcse-bert-base-uncased-sts")

if MODE == 'base':
    if not os.path.exists(f'wiki1m_for_simcse_tokenized.pickle'):
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

        with open(f'wiki1m_for_simcse_tokenized.pickle', 'wb') as f:
            pickle.dump(list_batch, f)
    else:
        with open(f'wiki1m_for_simcse_tokenized.pickle', 'rb') as f:
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
    
    embeddings = torch.stack(list_embeddings).reshape(-1, 768).detach().cpu().numpy()

    with open(f'wiki1m_for_simcse_embedding.pickle', 'wb') as f:
        pickle.dump(embeddings.tolist(), f)
        
elif MODE == 'sub':
    if not os.path.exists(f'wiki1m_for_simcse_tokenized_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle'):
        # load dataset
        with open(f'wiki1m_for_simcse_tree_cst_lg_subsentence.pickle', 'rb') as f:
            list_subsentence = pickle.load(f)

        # examples
        for subsentence in list_subsentence[:10]:
            print(subsentence)

        # make batch to load on GPU
        list_batch = []
        for subsentence in tqdm(list_subsentence):
            batch = tokenizer(text=subsentence, padding=True, truncation=True, return_tensors="pt", verbose=True)
            list_batch.append(batch)

        with open(f'wiki1m_for_simcse_tokenized_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'wb') as f:
            pickle.dump(list_batch, f)
    else:
        with open(f'wiki1m_for_simcse_tokenized_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'rb') as f:
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

    with open(f'wiki1m_for_simcse_embedding_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'wb') as f:
        pickle.dump(embeddings, f)