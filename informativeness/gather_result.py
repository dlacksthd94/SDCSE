import os
import pandas as pd
import re
from itertools import chain
import numpy as np

pd.set_option('display.max_rows', 150)

dict_plm = {
    'bert_base': 'bert-base-uncased',
    'bert_large': 'bert-large-uncased',
    'roberta_base': 'roberta-base',
    'roberta_large': 'roberta-large',
}

dict_lr = {
    'SDCSE': {
        'bert_base': '3e-5',
        'bert_large': '1e-5',
        'roberta_base': '1e-5',
        'roberta_large': '3e-5',
    },
    'DiffCSE': {
        'bert_base': '7e-6', # sts
        # 'bert_base': '2e-6', # transfer
    },
    'PromCSE': {
        'bert_base': '3e-2',
    },
    'MixCSE': {
        'bert_base': '3e-5',
    }
}

dict_meta = {    
    'SDCSE': {
        'list_plm': ['bert_base', 'bert_large', 'roberta_base', 'roberta_large'],
        'list_mode': ['fasttest', 'test'],
        'list_taskset': ['full'],
        'list_bs': [64, 128], # [64, 128, 256]
        'list_epoch': [1],
        'list_max_len': [32],
        'list_lambda_sdcse': ['0e-0', '2e-0', '5e-1'] + [f'1e-{i}' for i in range(0, 3)],
        'list_pt_type': ['mask_token', 'unk_token', 'pad_token', 'dropout', 'constituency_parsing', 'none'],
        'list_pt_num': [0, 1], # [0, 1, 2, 3]
        'list_pt_step': [0, 1, 2, 3, 4, 5, 1.5, 2.5, 3.5],
        'list_seed': range(0, 5),
        'list_loss': ['mse', 'margin'], # ['l1', 'sl1', 'mse', 'margin']
        'list_pooler': ['wp'], # ['wp', 'ap', 'wop']
        'list_metric': ['stsb'], # ['stsb', 'sickr', 'sts', 'transfer']
        'list_margin': ['0e-0', '1e-0', '1e-1'],
        'list_lambda_diffcse': ['0e-0'],
        'list_mask_ratio': ['0e-0'],
        'list_prom_len': [0],
    },
    
    'DiffCSE': {
        'list_plm': ['bert_base'],
        'list_mode': ['fasttest', 'test'],
        'list_taskset': ['full'],
        'list_bs': [64, 128],
        'list_epoch': [2],
        'list_max_len': [32],
        'list_lambda_sdcse': ['0e-0', '1e-0', '1e-1', '1e-2'],
        'list_pt_type': ['dropout', 'none'],
        'list_pt_num': [0, 1],
        'list_pt_step': [0, 1, 2, 3, 4],
        'list_seed': range(0, 5),
        'list_loss': ['mse', 'margin'],
        'list_pooler': ['wp'],
        'list_metric': ['stsb'],
        'list_margin': ['0e-0', '1e-0', '1e-1', '1e-2'],
        'list_lambda_diffcse': ['5e-3', '5e-2'],
        'list_mask_ratio': ['3e-1', '15e-2'],
        'list_prom_len': [0],
    },
    
    'PromCSE': {
        'list_plm': ['bert_base'],
        'list_mode': ['fasttest', 'test'],
        'list_taskset': ['full'],
        'list_bs': [256],
        'list_epoch': [1],
        'list_max_len': [32],
        'list_lambda_sdcse': ['0e-0', '1e-0', '1e-1', '1e-2', '5e-0'],
        'list_pt_type': ['dropout', 'none'],
        'list_pt_num': [0, 1],
        'list_pt_step': [0, 1, 2, 3, 4, 5],
        'list_seed': range(0, 5),
        'list_loss': ['mse', 'margin'],
        'list_pooler': ['wp'],
        'list_metric': ['stsb'],
        'list_margin': ['0e-0', '1e-0', '1e-1', '1e-2', '1e-3'],
        'list_lambda_diffcse': ['0e-0'],
        'list_mask_ratio': ['0e-0'],
        'list_prom_len': [16],
    },
        
    'MixCSE': {
        'list_plm': ['bert_base'],
        'list_mode': ['fasttest', 'test'],
        'list_taskset': ['full'],
        'list_bs': [64],
        'list_epoch': [1],
        'list_max_len': [32],
        'list_lambda_sdcse': ['0e-0', '1e-1', '1e-2'],
        'list_pt_type': ['dropout', 'none'],
        'list_pt_num': [0, 1],
        'list_pt_step': [0, 2],
        'list_seed': range(0, 5),
        'list_loss': ['mse', 'margin'],
        'list_pooler': ['wp'],
        'list_metric': ['stsb'],
        'list_margin': ['0e-0', '1e-1'],
        'list_lambda_diffcse': ['0e-0'],
        'list_mask_ratio': ['0e-0'],
        'list_prom_len': [0],
    },
}

list_sts = ['sts12', 'sts13', 'sts14', 'sts15', 'sts16', 'stsb', 'sickr']
list_transfer = ['mr', 'cr', 'subj', 'mpqa', 'sst2', 'trec', 'mrpc']

def result_dev(*groupby):
    root_path = os.path.join(os.path.expanduser('~'), 'PAPER/SDCSE/informativeness/', ENCODER, 'result', RESULT_FOLDER)
    list_result = []
    for plm in dict_meta[ENCODER]['list_plm']:
        for bs in dict_meta[ENCODER]['list_bs']:
            # for lr in [f'1e-{i}' for i in range(4, 6)] + ['3e-5']:
            for lr in [dict_lr[ENCODER][plm]]:
                for epoch in dict_meta[ENCODER]['list_epoch']:
                    for max_len in dict_meta[ENCODER]['list_max_len']:
                        for lambda_sdcse in dict_meta[ENCODER]['list_lambda_sdcse']:
                            for pt_type in dict_meta[ENCODER]['list_pt_type']:
                                for pt_num in dict_meta[ENCODER]['list_pt_num']:
                                    for pt_step in dict_meta[ENCODER]['list_pt_step']:
                                        for seed in dict_meta[ENCODER]['list_seed']:
                                            for loss in dict_meta[ENCODER]['list_loss']:
                                                for pooler in dict_meta[ENCODER]['list_pooler']:
                                                    for metric in dict_meta[ENCODER]['list_metric']:
                                                        for margin in dict_meta[ENCODER]['list_margin']:
                                                            for lambda_diffcse in dict_meta[ENCODER]['list_lambda_diffcse']:
                                                                for mask_ratio in dict_meta[ENCODER]['list_mask_ratio']:
                                                                    for prom_len in dict_meta[ENCODER]['list_prom_len']:
                                                                        
                                                                        try:
                                                                            # plm, bs, lr, epoch, seed, max_len, lambda_sdcse, pt_type, pt_num, pt_step, loss, pooler, metric, margin, lambda_diffcse, mask_ratio, prom_len = 'bert_base', 128, '3e-5', 1, 0, 32, '1e-0', 'dropout', 1, 2, 'margin', 'wp', 'stsb', '1e-1', '0e-0', '0e-0', 0 # SDCSE
                                                                            # plm, bs, lr, epoch, seed, max_len, lambda_sdcse, pt_type, pt_num, pt_step, loss, pooler, metric, margin, lambda_diffcse, mask_ratio, prom_len = 'bert_base', 64, '7e-6', 2, 0, 32, '0e-0', 'none', 0, 0, 'mse', 'wp', 'stsb', '0e-0', '5e-3', '3e-1', 0 # DiffCSE
                                                                            # plm, bs, lr, epoch, seed, max_len, lambda_sdcse, pt_type, pt_num, pt_step, loss, pooler, metric, margin, lambda_diffcse, mask_ratio, prom_len = 'bert_base', 256, '3e-2', 1, 0, 32, '0e-0', 'none', 0, 0, 'mse', 'wp', 'stsb', '0e-0', '0e-0', '0e-0', 16 # PromCSE
                                                                            result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-{dict_plm[plm]}_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_sdcse}_{pt_type}_{pt_num}_{pt_step}_{loss}_{pooler}_{metric}_{margin}_{lambda_diffcse}_{mask_ratio}_{prom_len}', 'eval_results.txt')
                                                                            if not os.path.exists(result_path):
                                                                                root_path_another = os.path.join('/data1/csl/', ENCODER, RESULT_FOLDER)
                                                                                result_path = os.path.join(root_path_another, f'my-unsup-{ENCODER.lower()}-{dict_plm[plm]}_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_sdcse}_{pt_type}_{pt_num}_{pt_step}_{loss}_{pooler}_{metric}_{margin}_{lambda_diffcse}_{mask_ratio}_{prom_len}', 'eval_results.txt')
                                                                            assert os.path.exists(result_path)
                                                                            df_temp = pd.read_csv(result_path, sep='=', header=None)
                                                                            df_temp = df_temp[-4:-2].reset_index(drop=True)
                                                                            df_temp.columns = ['task', 'score']
                                                                            result = [plm, bs, lr, epoch, max_len, lambda_sdcse, pt_type, pt_num, pt_step, loss, pooler, metric, margin, seed, lambda_diffcse, mask_ratio, prom_len, df_temp['score'][0] * 100, df_temp['score'][1]]
                                                                            list_result.append(result)
                                                                        except:
                                                                            pass
        
    df = pd.DataFrame(list_result, columns=['plm', 'bs', 'lr', 'epoch', 'max_len', 'lambda_sdcse', 'pt_type', 'pt_num', 'pt_step', 'loss', 'pooler', 'metric', 'margin', 'seed', 'lambda_diffcse', 'mask_ratio', 'prom_len', 'sts', 'transfer'])
    df_groupby = df.groupby([*groupby])['sts', 'transfer'].agg(['mean', 'std']).round(1)
    # if df_groupby.index.name == 'lambda_w':
    #     # print(df_groupby.loc[list_lambda_w])
    return df, df_groupby

def result_eval(*groupby):
    root_path = os.path.join(os.path.expanduser('~'), 'PAPER/SDCSE/informativeness/result/evaluation', ENCODER.lower(), RESULT_FOLDER)
    assert os.path.exists(root_path)
    dict_result = {}
    for plm in dict_meta[ENCODER]['list_plm']:
        for mode in dict_meta[ENCODER]['list_mode']:
            # for taskset in dict_meta[ENCODER]['list_taskset']:
            taskset = 'full'
            for bs in dict_meta[ENCODER]['list_bs']:
                # for lr in [f'1e-{i}' for i in range(4, 6)] + ['3e-5']:
                for lr in [dict_lr[ENCODER][plm]]:
                    for epoch in dict_meta[ENCODER]['list_epoch']:
                        # for max_len in dict_meta[ENCODER]['list_max_len']:
                        max_len = 32
                        for lambda_sdcse in dict_meta[ENCODER]['list_lambda_sdcse']:
                            for pt_type in dict_meta[ENCODER]['list_pt_type']:
                                for pt_num in dict_meta[ENCODER]['list_pt_num']:
                                    for pt_step in dict_meta[ENCODER]['list_pt_step']:
                                        for seed in dict_meta[ENCODER]['list_seed']:
                                            for loss in dict_meta[ENCODER]['list_loss']:
                                                for pooler in dict_meta[ENCODER]['list_pooler']:
                                                    for metric in dict_meta[ENCODER]['list_metric']:
                                                        for margin in dict_meta[ENCODER]['list_margin']:
                                                            for lambda_diffcse in dict_meta[ENCODER]['list_lambda_diffcse']:
                                                                for mask_ratio in dict_meta[ENCODER]['list_mask_ratio']:
                                                                    for prom_len in dict_meta[ENCODER]['list_prom_len']:
                                                                        try:
                                                                            # plm, mode, taskset, bs, lr, epoch, seed, max_len, lambda_sdcse, pt_type, pt_num, pt_step, loss, pooler, metric, margin, lambda_diffcse, mask_ratio, prom_len = 'bert_base', 'test', 'full', 64, '7e-6', 2, 0, 32, '0e-0', 'none', 0, 0, 'mse', 'wp', 'stsb', '0e-0', '5e-3', '3e-1', 0 # DiffCSE
                                                                            # plm, mode, taskset, bs, lr, epoch, seed, max_len, lambda_sdcse, pt_type, pt_num, pt_step, loss, pooler, metric, margin, lambda_diffcse, mask_ratio, prom_len = 'bert_base', 'test', 'full', 256, '3e-2', 1, 0, 32, '1e-1', 'dropout', 1, 2, 'margin', 'wp', 'stsb', '1e-1', '0e-0', '0e-0', 16 # PromCSE
                                                                            # plm, mode, taskset, bs, lr, epoch, seed, max_len, lambda_sdcse, pt_type, pt_num, pt_step, loss, pooler, metric, margin, lambda_diffcse, mask_ratio, prom_len = 'bert_base', 'test', 'full', 64, '3e-5', 1, 0, 32, '1e-2', 'dropout', 1, 2, 'margin', 'wp', 'stsb', '1e-1', '0e-0', '0e-0', 0 # MixCSE
                                                                            if ENCODER == 'SimCSE':
                                                                                result_path = os.path.join(root_path, f'{mode}_{taskset}_unsup_{ENCODER.lower()}_{plm}_{bs}_{lr}_{epoch}_{seed}_{max_len}_{pooler}_{metric}_{margin}_{prom_len}.txt')
                                                                            else:
                                                                                result_path = os.path.join(root_path, f'{mode}_{taskset}_unsup_{ENCODER.lower()}_{plm}_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_sdcse}_{pt_type}_{pt_num}_{pt_step}_{loss}_{pooler}_{metric}_{margin}_{lambda_diffcse}_{mask_ratio}_{prom_len}.txt')
                                                                            assert os.path.exists(result_path)                                                                                
                                                                            with open(result_path, 'r') as f:
                                                                                text = f.read()
                                                                            text = re.sub(r'-* fasttest -*', '', text).strip()
                                                                            text = re.sub(r'-* test -*', '', text).strip()
                                                                            index_split = text.find('+\n+')
                                                                            text1 = text[:index_split + 1]
                                                                            text2 = text[index_split + 1:].strip()
                                                                            
                                                                            def prettytable_to_dataframe(text):
                                                                                text = text.split('\n')
                                                                                while not text[0].startswith('+-------+'):
                                                                                    del text[0]
                                                                                
                                                                                # 헤더 추출
                                                                                headers = text[1].strip().split('|')[1:-1]
                                                                                headers = [header.strip() for header in headers]
                                                                                # 데이터 추출
                                                                                data = []
                                                                                for line in text[3:]:
                                                                                    if line.startswith('+') and line.endswith('+'):
                                                                                        continue
                                                                                    row = line.strip().split('|')[1:-1]
                                                                                    row = [value.strip() for value in row]
                                                                                    data.append(row)
                                                                                df = pd.DataFrame(data, columns=headers)
                                                                                return df

                                                                            df1 = prettytable_to_dataframe(text1)
                                                                            list1 = df1.squeeze().to_list()
                                                                            if text2:
                                                                                df2 = prettytable_to_dataframe(text2)
                                                                                list2 = df2.squeeze().to_list()
                                                                                list_score = list1[:-1] + list2[:-1] + [list1[-1], list2[-1]]
                                                                            else:
                                                                                list_score = list1[:-1] + [0] * 7 + [list1[-1], 0]
                                                                            result = [mode, taskset, plm, bs, lr, epoch, max_len, lambda_sdcse, pt_type, pt_num, pt_step, loss, pooler, metric, margin, lambda_diffcse, mask_ratio, prom_len, seed] + list(map(float, list_score))
                                                                            dict_result[result_path] = result
                                                                        except:
                                                                            pass

    df = pd.DataFrame(dict_result.values(), columns=['mode', 'taskset', 'plm', 'bs', 'lr', 'epoch', 'max_len', 'lambda_sdcse', 'pt_type', 'pt_num', 'pt_step', 'loss', 'pooler', 'metric', 'margin', 'lambda_diffcse', 'mask_ratio', 'prom_len', 'seed'] + list_sts + list_transfer + ['sts', 'transfer'])
    df_groupby = df.groupby([*groupby]).agg([np.mean, np.std])
    # if df_groupby.index.name == 'lambda_w':
    #     # print(df_groupby.loc[list_lambda_w])
    return df, df_groupby

# ENCODER = 'SDCSE'
# # RESULT_FOLDER = 'backup_eval_dropout_sim1_all'
# # RESULT_FOLDER = 'backup_eval_dropout_sim1_nocls'
# # RESULT_FOLDER = 'backup_eval_dropout_sim0_nocls'
# # RESULT_FOLDER = 'backup_eval_dropout_sim0_all'
# RESULT_FOLDER = 'backup_eval_dropout_sim0_nocls_1gpu'

# ENCODER = 'DiffCSE'
# # RESULT_FOLDER = 'backup_eval_dropout_sim0_nocls_sts'
# # RESULT_FOLDER = 'backup_eval_dropout_sim0_nocls_transfer'
# RESULT_FOLDER = 'backup_eval_dropout_sim0_nocls_sts_1gpu'
# # RESULT_FOLDER = 'backup_eval_dropout_sim0_all_sts_1gpu'
# # RESULT_FOLDER = 'backup_eval_dropout_sim0_nocls_sts_nobn_1gpu'

# ENCODER = 'PromCSE'
# RESULT_FOLDER = 'backup_eval_dropout_sim0_nocls_1gpu'

ENCODER = 'MixCSE'
RESULT_FOLDER = 'backup_eval_dropout_sim0_nocls_1gpu'

result_dev('plm', 'bs', 'loss', 'lambda_sdcse', 'pt_type', 'pooler', 'pt_num', 'pt_step', 'margin', 'lambda_diffcse', 'mask_ratio', 'prom_len')[1]
result_eval('mode', 'taskset', 'plm', 'bs', 'loss', 'lambda_sdcse', 'pt_type', 'pooler', 'pt_num', 'pt_step', 'margin', 'lambda_diffcse', 'mask_ratio', 'prom_len')[1].loc[:, ['sts', 'transfer']].round(1)

x = result_eval('mode', 'taskset', 'plm', 'bs', 'loss', 'lambda_w', 'pt_type', 'pooler', 'pt_num', 'pt_step', 'margin')[1].loc[:, list_sts + ['sts']]
x[pd.MultiIndex.from_product([list_sts + ['sts'], ['std']])] = x[pd.MultiIndex.from_product([list_sts + ['sts'], ['std']])] * 1.96 / np.sqrt(5)
x = x.round(2).apply(lambda x: [format(xi, ".1f") for xi in x])
# for col in list_sts + ['sts']:
for col in ['sts']:
    x.loc[:, (col, 'mean')] = r'{' + x.loc[:, (col, 'mean')] + r'}$_{\pm' + x.loc[:, (col, 'std')] + '}$ &'
x[pd.MultiIndex.from_product([list_sts + ['sts'], ['mean']])]


x = result_eval('mode', 'taskset', 'plm', 'bs', 'loss', 'lambda_w', 'pt_type', 'pooler', 'pt_num', 'pt_step', 'margin')[1].loc[:, list_transfer + ['transfer']]
x[pd.MultiIndex.from_product([list_transfer + ['transfer'], ['std']])] = x[pd.MultiIndex.from_product([list_transfer + ['transfer'], ['std']])] * 1.96 / np.sqrt(5)
x = x.round(2).apply(lambda x: [format(xi, ".1f") for xi in x])
# for col in list_transfer + ['transfer']:
for col in ['transfer']:
    x.loc[:, (col, 'mean')] = r'{' + x.loc[:, (col, 'mean')] + r'}$_{\pm' + x.loc[:, (col, 'std')] + '}$ &'
x[pd.MultiIndex.from_product([list_transfer + ['transfer'], ['mean']])]
