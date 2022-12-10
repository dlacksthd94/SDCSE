import pickle
import nltk
from tqdm import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-d', required=False, help='select dataset from [wiki1m, STS12, STS13, STS14, STS15, STS16, STS-B, SICK-R, nli, quora, simplewiki, specter, covid]', type=str, choices=['wiki1m', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R', 'nli', 'quora', 'simplewiki', 'specter', 'covid'], default='wiki1m')

DATASET = parser.parse_args().d
print(DATASET)

##### mask sentences
with open(f'../SimCSE/{DATASET + "_for_simcse" if DATASET == "wiki1m" else DATASET}.txt') as f:
    list_text = f.readlines()

list_masked_sentence = []
for text in tqdm(list_text):
    list_masked_sentence_temp = []
    list_token = text.strip().split()
    if len(list_token) > 1:
        list_masked_sentence_temp.append(text)
        list_idx2mask = np.random.choice(range(len(list_token)), 5)
        for idx2mask in list_idx2mask:
            list_token_temp = list_token[:]
            list_token_temp[idx2mask] = '[MASK]'
            masked_sentence = ' '. join(list_token_temp)
            list_masked_sentence_temp.append(masked_sentence)
        list_masked_sentence.append(list_masked_sentence_temp)
    else:
        list_masked_sentence.append([text] * 6)

with open(f'data/{DATASET}_masked.pickle', 'wb') as f:
    pickle.dump(list_masked_sentence, f)