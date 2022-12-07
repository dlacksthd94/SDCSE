import spacy
from spacy import displacy
from tqdm import tqdm
import joblib as jl
import pickle
import numpy as np
import os

# load dataset
with open('../SimCSE/wiki1m_for_simcse.txt') as f:
    list_text = f.readlines()

# preprocessing
for i in tqdm(range(len(list_text))):
    list_text[i] = list_text[i].strip('\n.')

if not os.path.exists(f'wiki1m_for_simcse_tree_dpd.pickle'):
    # Load the language model
    model = spacy.load("en_core_web_sm")

    # parse sentence
    list_tree = []
    for text in tqdm(list_text):
        doc = model(text)
        list_tree.append(doc)

    with open(f'wiki1m_for_simcse_tree_dpd.pickle', 'wb') as f:
        pickle.dump(list_tree, f)
else:    
    with open(f'wiki1m_for_simcse_tree_dpd.pickle', 'rb') as f:
        list_tree = pickle.load(f)

# get depth
def walk_tree_dpd(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree_dpd(child, depth + 1) for child in node.children)
    else:
        return depth

doc = list_tree[15]
list_depth = []
for doc in tqdm(list_tree):
    # nlp function returns an object with individual token information, 
    # linguistic features and relationships

    # print ("{:<15} | {:<8} | {:<15} | {:<20}".format('Token','Relation','Head', 'Children'))
    # print ("-" * 70)
    # for token in doc:
    #   # Print the token, dependency nature, head and all dependents of the token
    #   print ("{:<15} | {:<8} | {:<15} | {:<20}"
    #          .format(str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])))

    if len(doc):
        depth = walk_tree_dpd(list(doc.sents)[0].root, 0)
        # [walk_tree_dpd(sent.root, 0) for sent in doc.sents]
    else:
        depth = 0
    list_depth.append(depth)
    # return depth

# # Use displayCy to visualize the dependency 
# displacy.render(doc, style='dep', jupyter=True, options={'distance': 120})

with open(f'wiki1m_for_simcse_tree_dpd_depth.pickle', 'wb') as f:
    pickle.dump(list_depth, f)