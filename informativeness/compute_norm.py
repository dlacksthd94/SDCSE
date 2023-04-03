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
parser.add_argument('-n', required=False, help='select normalization method from [orig, zero_mean, zscore]', type=str, choices=['orig', 'zero_mean', 'zscore'], default='orig')

PARSER = 'benepar_en3' if parser.parse_args().p == 'base' else 'benepar_en3_large'
PIPELINE = f'en_core_web_{parser.parse_args().pp}'
MODE = parser.parse_args().m
N = ''
NORMALIZATION = parser.parse_args().n

# list_dataset = ['wiki1m', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R', 'quora', 'simplewiki', 'specter', 'covid', 'huffpost']
list_dataset = ['wiki1m']
list_encoder = ['bert', 'sbert', 'simcse', 'diffcse', 'promcse', 'mcse']
list_plm = ['bert', 'roberta']

df_result = pd.DataFrame(columns=[list(chain.from_iterable(zip(list_encoder, list_encoder))), list_plm * len(list_encoder)], index=list_dataset)
list_num_data = []
for dataset in tqdm(list_dataset):
    for plm in tqdm(list_plm, leave=False):
        for encoder in tqdm(list_encoder, leave=False):
            with open(f'data/{dataset}_{encoder}_{plm}_embedding_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'rb') as f:
                list_embedding = pickle.load(f)
            list_embedding_first = list(map(lambda x: x[0], list_embedding))
            num_data = np.stack(list_embedding).shape[0]
            embedding_avg = np.concatenate(list_embedding).mean(axis=0)
            embedding_std = np.concatenate(list_embedding).std(axis=0)
            
            # # show example
            # dataset = 'simplewiki'
            # encoder = 'simcse'
            # plm = 'bert'
            # with open(f'data/{dataset}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}{N}_subsentence.pickle', 'rb') as f:
            #     list_subsentences = pickle.load(f)
            # # list_i = 33132, 493763, 695194, 908158, 225215, 514394, 848055, 901159, 348773, 736685, 544060, 643723
            # i = np.random.randint(0, len(list_embedding))
            # show_example(list_embedding, list_subsentences, i)
            
            list_corr = []
            list_len = []
            for i in tqdm(range(len(list_embedding)), leave=False):
                if list_embedding[i].shape[0] >= 2:
                    if NORMALIZATION == 'orig':
                        embedding_temp = list_embedding[i][:]
                    elif NORMALIZATION == 'zero_mean':
                        embedding_temp = list_embedding[i][:] - embedding_avg
                    elif NORMALIZATION == 'zscore':
                        embedding_temp = (list_embedding[i][:] - embedding_avg) / embedding_std
                    list_norm_temp = np.linalg.norm(embedding_temp, axis=1)
                    list_norm_rank = list_norm_temp.argsort()[::-1]
                    corr_s = spearmanr(list_norm_rank, np.arange(list_norm_temp.shape[0]))[0]
                    list_corr.append(corr_s)
                    list_len.append(list_norm_temp.shape[0])
            corr_wmean = (np.array(list_corr) * np.array(list_len) / np.array(list_len).sum()).sum()
            df_result[encoder, plm][dataset] = corr_wmean
    list_num_data.append(num_data)

avg = df_result.mean(axis=0)
avg_w = df_result.apply(lambda col: col * list_num_data).sum(axis=0) / sum(list_num_data)
df_result.loc['avg'] = avg
df_result.loc['avg_w'] = avg_w
df_result = df_result.astype(float).round(3)
df_result.to_csv(f'result/result_subsentence_norm_{PIPELINE[12:]}{PARSER[11:]}{N}_{NORMALIZATION}.csv')
