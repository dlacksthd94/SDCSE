import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import spearmanr, pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('-p', required=False, help='select constituency parser from [base, large]', type=str, choices=['base', 'large'], default='base')
parser.add_argument('-pp', required=False, help='select constituency parser pipeline [sm, md, lg]', type=str, choices=['sm', 'md', 'lg'], default='lg')
parser.add_argument('-m', required=False, help='select mode from [base, sub, mask]', type=str, choices=['base', 'sub', 'mask'], default='sub')

PARSER = 'benepar_en3' if parser.parse_args().p == 'base' else 'benepar_en3_large'
PIPELINE = f'en_core_web_{parser.parse_args().pp}'
MODE = parser.parse_args().m
N = ''

# list_dataset = ['wiki1m', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R', 'nli', 'quora', 'simplewiki', 'specter', 'covid']
list_dataset = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R', 'nli', 'quora', 'simplewiki', 'specter', 'covid']
list_encoder = ['bert', 'sbert', 'simcse', 'diffcse']

def show_example(list_embedding, list_subsentences, i):
    list_norm_temp = (list_embedding[i] ** 2).sum(axis=1) ** 0.5    
    # check norm with sentence
    pd.set_option('display.max_colwidth', None)
    list_subsentence = list_subsentences[i]
    df_exmample = pd.DataFrame({'text': list_subsentence, 'norm': list_norm_temp})
    df_exmample['norm_diff'] = df_exmample['norm'] - df_exmample['norm'][0]
    df_exmample['cos_sim'] = cosine_similarity(list_embedding[i])[0]
    df_exmample['l2_dis'] = euclidean_distances(list_embedding[i])[0]
    df_exmample['l2_dis_norm'] = euclidean_distances(np.divide(list_embedding[i], list_norm_temp.reshape(-1, 1)))[0]
    df_exmample = df_exmample.round(2)
    return df_exmample

# computing subsentence norm
if MODE == 'sub':
    
    df_result = pd.DataFrame(columns=list_encoder, index=list_dataset)
    for dataset in tqdm(list_dataset):
        # dataset = 'STS12'
        # encoder = 'simcse'
        with open(f'data/{dataset}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}{N}_subsentence.pickle', 'rb') as f:
            list_subsentences = pickle.load(f)
        for encoder in tqdm(list_encoder, leave=False):
            with open(f'data/{dataset}_{encoder}_embedding_subsentence_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'rb') as f:
                list_embedding = pickle.load(f)
            
            # show example
            # list_i = 33132, 493763, 695194, 908158, 225215, 514394, 848055, 901159, 348773, 736685, 544060, 643723
            i = np.random.randint(0, len(list_embedding))
            show_example(list_embedding, list_subsentences, i)
            
            list_corr = []
            list_len = []
            for i in tqdm(range(len(list_embedding)), leave=False):
                if list_embedding[i].shape[0] >= 2:
                    list_norm_temp = (list_embedding[i] ** 2).sum(axis=1) ** 0.5
                    list_norm_rank = list_norm_temp.argsort()[::-1]
                    corr_s = spearmanr(list_norm_rank, np.arange(list_norm_temp.shape[0]))[0]
                    list_corr.append(corr_s)
                    list_len.append(list_norm_temp.shape[0])
            corr_wmean = (np.array(list_corr) * np.array(list_len) / np.array(list_len).sum()).sum()
            df_result[encoder][dataset] = corr_wmean
    df_result.loc['avg'] = df_result.mean(axis=0)
    df_result.to_csv('result_subsentence.csv')

elif MODE == 'mask':
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