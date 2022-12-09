import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-p', required=False, help='select constituency parser from [base, large]', type=str, choices=['base', 'large'], default='base')
parser.add_argument('-pp', required=False, help='select constituency parser pipeline [sm, md, lg]', type=str, choices=['sm', 'md', 'lg'], default='lg')

PARSER = 'benepar_en3' if parser.parse_args().p == 'base' else 'benepar_en3_large'
PIPELINE = f'en_core_web_{parser.parse_args().pp}'
N = ''

list_dataset = ['wiki1m', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R']
list_encoder = ['bert', 'sbert', 'simcse', 'diffcse']

# computing subsentence norm
df_result = pd.DataFrame(columns=list_encoder, index=list_dataset)
for dataset in tqdm(list_dataset):
    for encoder in tqdm(list_encoder, leave=False):
        with open(f'data/{dataset}_{encoder}_embedding_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'rb') as f:
            list_embedding = pickle.load(f)
        with open(f'data/{dataset}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}{N}_subsentence.pickle', 'rb') as f:
            list_subsentences = pickle.load(f)
        
        list_norm_diff_sum = []
        for i in tqdm(range(len(list_embedding)), leave=False):
            # i = np.random.choice(range(len(list_embedding)), 1)[0]
            list_norm = (list_embedding[i] ** 2).sum(axis=1) ** 0.5
            if len(list_norm) >= 2:
                norm_diff_sum = (list_norm[0:-1] - list_norm[1:]).sum()
            # list_subsentence = list_subsentences[i]
            # for (norm, subsentence) in zip(list_norm, list_subsentence):
            #     print(norm, '\t', subsentence)
            # pd.DataFrame({'text': list_subsentence, 'norm': list_norm})
            list_norm_diff_sum.append(norm_diff_sum)
        norm_diff_mean = np.array(list_norm_diff_sum).mean()
        df_result[encoder][dataset] = norm_diff_mean

df_result.to_csv('result_subsentence.csv')