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

# list_dataset = ['wiki1m', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R']
# list_dataset = ['covid', 'huffpost']
list_dataset = ['wiki1m']
list_encoder = ['bert', 'sbert', 'simcse', 'diffcse', 'promcse']
list_plm = ['bert', 'roberta']
list_score = ['mean', 'std', '>0.2', '>0.4', '>0.6']

df_result = pd.DataFrame(columns=[list(chain.from_iterable(zip(*[list_encoder] * len(list_plm) * len(list_score)))), list(chain.from_iterable(zip(*[list_plm] * len(list_score)))) * len(list_encoder), list_score * len(list_encoder) * len(list_plm)], index=list_dataset)
list_num_data = []
for dataset in tqdm(list_dataset):
    for plm in tqdm(list_plm, leave=False):
        for encoder in tqdm(list_encoder, leave=False):
            with open(f'data/{dataset}_{encoder}_{plm}_embedding_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'rb') as f:
                list_embedding = pickle.load(f)
            plm, encoder, np.concatenate(list_embedding).shape
            
            num_data = np.concatenate(list_embedding).shape[0]
            
            embedding_avg = np.concatenate(list_embedding).mean(axis=0)
            embedding_avg_total = embedding_avg.mean()
            embedding_std = np.concatenate(list_embedding).std(axis=0)
            embedding_std_total = embedding_std.mean()
            over_2 = sum(abs(embedding_avg) >= 0.2)
            over_4 = sum(abs(embedding_avg) >= 0.4)
            over_6 = sum(abs(embedding_avg) >= 0.6)

            df_result[encoder, plm, 'mean'][dataset] = embedding_avg_total
            df_result[encoder, plm, 'std'][dataset] = embedding_std_total
            df_result[encoder, plm, '>0.2'][dataset] = over_2
            df_result[encoder, plm, '>0.4'][dataset] = over_4
            df_result[encoder, plm, '>0.6'][dataset] = over_6
    list_num_data.append(num_data)

avg = df_result.mean(axis=0)
avg_w = df_result.apply(lambda col: col * list_num_data).sum(axis=0) / sum(list_num_data)
df_result.loc['avg'] = avg
df_result.loc['avg_w'] = avg_w
df_result = df_result.astype(float).round(3)
df_result = df_result.loc['avg_w':].stack().droplevel(0)
df_result = df_result[list_encoder]
df_result = df_result.loc[list_score]
df_result = df_result.T
df_result = df_result.astype({'>0.2': int, '>0.4': int, '>0.6': int})
df_result.to_csv(f'result/result_mean_std_{PIPELINE[12:]}{PARSER[11:]}{N}.csv')