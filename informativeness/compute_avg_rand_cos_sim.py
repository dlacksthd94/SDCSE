import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import spearmanr, pearsonr
from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument('-p', required=False, help='select constituency parser from [base, large]', type=str, choices=['base', 'large'], default='large')
parser.add_argument('-pp', required=False, help='select constituency parser pipeline [sm, md, lg]', type=str, choices=['sm', 'md', 'lg'], default='lg')
parser.add_argument('-m', required=False, help='select mode from [base, sub, mask]', type=str, choices=['base', 'sub', 'mask'], default='sub')

PARSER = 'benepar_en3' if parser.parse_args().p == 'base' else 'benepar_en3_large'
PIPELINE = f'en_core_web_{parser.parse_args().pp}'
MODE = parser.parse_args().m
N = ''

# list_dataset = ['wiki1m', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R', 'quora', 'simplewiki', 'specter', 'covid', 'huffpost']
# list_dataset = ['covid', 'huffpost']
list_dataset = ['wiki1m']
list_encoder = ['bert', 'sbert', 'simcse', 'diffcse', 'promcse']
list_plm = ['bert', 'roberta']

df_result = pd.DataFrame(columns=[list(chain.from_iterable(zip(list_encoder, list_encoder))), list_plm * len(list_encoder)], index=list_dataset)
list_num_data = []
for dataset in tqdm(list_dataset):
    for plm in tqdm(list_plm, leave=False):
        for encoder in tqdm(list_encoder, leave=False):
            with open(f'data/{dataset}_{encoder}_{plm}_embedding_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'rb') as f:
                list_embedding = pickle.load(f)
            num_data = np.concatenate(list_embedding).shape[0]
            embedding_original_concat = np.stack(list(map(lambda x: x[0], list_embedding)))
            
            cos_sim_sum = 0
            for i in tqdm(range(10000), leave=False):
                a, b = np.random.choice(embedding_original_concat.shape[0], 2, replace=False)
                cos_sim_temp = cosine_similarity(embedding_original_concat[[a, b]])[0, 1]
                cos_sim_sum += cos_sim_temp
            cos_sim_avg = cos_sim_sum / 10000

            df_result[encoder, plm][dataset] = cos_sim_avg
    list_num_data.append(num_data)

avg = df_result.mean(axis=0)
avg_w = df_result.apply(lambda col: col * list_num_data).sum(axis=0) / sum(list_num_data)
df_result.loc['avg'] = avg
df_result.loc['avg_w'] = avg_w
df_result = df_result.astype(float).round(3)
df_result.to_csv(f'result/result_avg_rand_cos_sim_{PIPELINE[12:]}{PARSER[11:]}{N}.csv')