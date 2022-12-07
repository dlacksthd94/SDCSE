import spacy
from spacy import displacy
from tqdm import tqdm
import joblib as jl
import pickle
import numpy as np
import os
import benepar
import argparse
import torch

# arg
parser = argparse.ArgumentParser()
parser.add_argument('-n', required=False, help='-1: all, 0: 1/4, 1: 2/4, 2: 3/4, 3: 4/4', type=int, choices=[0, 1, 2, 3, 4], default=-1)

MODEL = 'benepar_en3'
# MODEL = 'benepar_en3_large'
# PIPELINE = 'en_core_web_sm'
# PIPELINE = 'en_core_web_md'
PIPELINE = 'en_core_web_lg'
N = '_' + str(parser.parse_args().n + 1) if parser.parse_args().n + 1 else ''
START_INDEX = parser.parse_args().n * 250000 if parser.parse_args().n + 1 else 0
print(N)
print(START_INDEX)

# set gpu id
torch.cuda.set_device((parser.parse_args().n + 2) // 2)
print(torch.cuda.current_device())

# load dataset
with open('../SimCSE/wiki1m_for_simcse.txt') as f:
    list_text = f.readlines()

# preprocessing
for i in tqdm(range(len(list_text))):
    list_text[i] = list_text[i].strip('\n.')

# load model
benepar.download(MODEL)
model = spacy.load(PIPELINE)

if spacy.__version__.startswith('2'):
    model.add_pipe(benepar.BeneparComponent(MODEL))
else:
    model.add_pipe("benepar", config={"model": MODEL})

# start parsing
if not os.path.exists(f'wiki1m_for_simcse_tree_cst_{PIPELINE[12:]}{MODEL[11:]}{N}.pickle'):
    # parse sentence
    list_tree = []
    for text in tqdm(list_text[START_INDEX:START_INDEX + 250000], initial=START_INDEX, total=1000000):
        try:
            doc = model(text)
            list_tree.append(doc)
        except:
            list_tree.append([])

    with open(f'wiki1m_for_simcse_tree_cst_{PIPELINE[12:]}{MODEL[11:]}{N}.pickle', 'wb') as f:
        pickle.dump(list_tree, f)
else:    
    with open(f'wiki1m_for_simcse_tree_cst_{PIPELINE[12:]}{MODEL[11:]}{N}.pickle', 'rb') as f:
        list_tree = pickle.load(f)

# get depth
def walk_tree_cst(node, depth):
    if list(node._.children):
        return max([walk_tree_cst(child, depth + 1) for child in node._.children])
    else:
        return depth

list_depth = []
for doc in tqdm(list_tree):
    
    # print ("{:<15} | {:<8} | {:<15} | {:<20}".format('Token','Relation','Head', 'Children'))
    # print ("-" * 70)
    # for token in doc:
    #   # Print the token, dependency nature, head and all dependents of the token
    #   print ("{:<15} | {:<8} | {:<15} | {:<20}"
    #          .format(str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])))

    if len(doc):
        depth = walk_tree_cst(list(doc.sents)[0], 0)
        # [walk_tree_cst(sent.root, 0) for sent in doc.sents]
    else:
        depth = 0
    list_depth.append(depth)

# # Use displayCy to visualize the dependency 
# displacy.render(doc, style='dep', jupyter=True, options={'distance': 120})

with open(f'wiki1m_for_simcse_tree_cst_{PIPELINE[12:]}{MODEL[11:]}{N}_depth.pickle', 'wb') as f:
    pickle.dump(list_depth, f)