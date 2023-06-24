import os
import pandas as pd
import re
from itertools import chain
import numpy as np

pd.set_option('display.max_rows', 100)

dict_plm = {
    'bert_base': 'bert-base-uncased',
    'bert_large': 'bert-large-uncased',
    'roberta_base': 'roberta-base',
    'roberta_large': 'roberta-large',
}

dict_lr = {
    'bert_base': '3e-5',
    'bert_large': '1e-5',
    'roberta_base': '1e-5',
    'roberta_large': '3e-5',
}

list_sts = ['sts12', 'sts13', 'sts14', 'sts15', 'sts16', 'stsb', 'sickr']
list_transfer = ['mr', 'cr', 'subj', 'mpqa', 'sst2', 'trec', 'mrpc']

def result_dev(*groupby):
    root_path = os.path.join(os.path.expanduser('~'), 'PAPER/SDCSE/informativeness/', ENCODER, 'result', RESULT_FOLDER)
    
    list_result = []
    for plm in ['bert_base', 'bert_large', 'roberta_base', 'roberta_large']:
        # for bs in [64, 128, 256]:
        for bs in [64, 128]:
            # for lr in [f'1e-{i}' for i in range(4, 6)] + ['3e-5']:
            for lr in [dict_lr[plm]]:
                for epoch in range(1, 2):
                    for max_len in [32]:
                        list_lambda_w = ['0e-0', '2e-0', '5e-1'] + [f'1e-{i}' for i in range(0, 2)]
                        for lambda_w in list_lambda_w:
                            for pt_type in ['mask_token', 'unk_token', 'pad_token', 'dropout', 'none']:
                            # for pt_type in ['mask_token', 'dropout', 'none']:
                                # for pt_num in [0, 1, 2, 3]:
                                for pt_num in [0, 1]:
                                    for pt_step in [0, 1, 2, 3, 4, 5]:
                                        for seed in range(0, 5):
                                            # for loss in ['l1', 'sl1', 'mse', 'margin']:
                                            for loss in ['margin', 'mse']:
                                                # for pooler in ['wp', 'ap', 'wop']:
                                                for pooler in ['wp']:
                                                    # for metric in ['stsb', 'sickr', 'sts', 'transfer']:for metric in ['stsb', 'sickr', 'sts', 'transfer']:
                                                    for metric in ['stsb']:
                                                        for margin in ['0e-0', '1e-0', '1e-1']:
                                                            try:
                                                                result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-{dict_plm[plm]}_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_w}_{pt_type}_{pt_num}_{pt_step}_{loss}_{pooler}_{metric}_{margin}', 'eval_results.txt')
                                                                if not os.path.exists(result_path):
                                                                    root_path_another = os.path.join(os.path.expanduser('~'), 'PAPER/SDCSE/informativeness/', ENCODER, 'result')
                                                                    result_path = os.path.join(root_path_another, f'my-unsup-{ENCODER.lower()}-{dict_plm[plm]}_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_w}_{pt_type}_{pt_num}_{pt_step}_{loss}_{pooler}_{metric}_{margin}', 'eval_results.txt')
                                                                if not os.path.exists(result_path):
                                                                    root_path_another = os.path.join('/data1/csl/', RESULT_FOLDER)
                                                                    result_path = os.path.join(root_path_another, f'my-unsup-{ENCODER.lower()}-{dict_plm[plm]}_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_w}_{pt_type}_{pt_num}_{pt_step}_{loss}_{pooler}_{metric}_{margin}', 'eval_results.txt')
                                                                # if not os.path.exists(result_path):
                                                                #     result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-{dict_plm[plm]}_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_w}', 'eval_results.txt')
                                                                # if not os.path.exists(result_path):
                                                                #     result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-{dict_plm[plm]}_{bs}_{lr}_{epoch}_{seed}_{max_len}', 'eval_results.txt')
                                                                # if not os.path.exists(result_path):
                                                                #     result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-{dict_plm[plm]}_{bs}_{lr}_{epoch}_{seed}', 'eval_results.txt')
                                                                assert os.path.exists(result_path)
                                                                df_temp = pd.read_csv(result_path, sep='=', header=None)
                                                                df_temp = df_temp[-4:-2].reset_index(drop=True)
                                                                df_temp.columns = ['task', 'score']
                                                                result = [plm, bs, lr, epoch, max_len, lambda_w, pt_type, pt_num, pt_step, loss, pooler, metric, margin, seed, round(df_temp['score'][0] * 100, 2), round(df_temp['score'][1], 2)]
                                                                list_result.append(result)
                                                            except:
                                                                pass
    df = pd.DataFrame(list_result, columns=['plm', 'bs', 'lr', 'epoch', 'max_len', 'lambda_w', 'pt_type', 'pt_num', 'pt_step', 'loss', 'pooler', 'metric', 'margin', 'seed', 'sts', 'transfer'])
    df_groupby = df.groupby([*groupby])['sts', 'transfer'].agg(['mean', 'std'])
    # if df_groupby.index.name == 'lambda_w':
    #     # print(df_groupby.loc[list_lambda_w])
    return df, df_groupby

def result_eval(*groupby):
    root_path = os.path.join(os.path.expanduser('~'), 'PAPER/SDCSE/informativeness/result/evaluation', ENCODER.lower(), RESULT_FOLDER)
    assert os.path.exists(root_path)

    dict_result = {}
    for plm in ['bert_base', 'bert_large', 'roberta_base', 'roberta_large']:
        for mode in ['fasttest', 'test']:
            for taskset in ['full']:
                # for bs in [64, 128, 256]:
                for bs in [64, 128]:
                    # for lr in [f'1e-{i}' for i in range(4, 6)] + ['3e-5']:
                    for lr in [dict_lr[plm]]:
                        for epoch in range(1, 2):
                            for max_len in [32]:
                                list_lambda_w = ['0e-0', '2e-0', '5e-1'] + [f'1e-{i}' for i in range(0, 2)]
                                for lambda_w in list_lambda_w:
                                    for pt_type in ['mask_token', 'unk_token', 'pad_token', 'dropout', 'none']:
                                    # for pt_type in ['mask_token', 'dropout', 'none']:
                                        for pt_num in [0, 1]:
                                            for pt_step in [0, 1, 2, 3, 4, 5]:
                                                for seed in range(0, 5):
                                                    # for loss in ['l1', 'sl1', 'mse', 'margin']:
                                                    for loss in ['margin', 'mse']:
                                                        # for pooler in ['wp', 'ap', 'wop']:
                                                        for pooler in ['wp']:
                                                            # for metric in ['stsb', 'sickr', 'sts', 'transfer']:
                                                            for metric in ['stsb']:
                                                                for margin in ['0e-0', '1e-0', '1e-1']:
                                                                    try:
                                                                        if ENCODER == 'SimCSE':
                                                                            result_path = os.path.join(root_path, f'{mode}_{taskset}_unsup_{ENCODER.lower()}_{plm}_{bs}_{lr}_{epoch}_{seed}_{max_len}_{pooler}_{metric}_{margin}.txt')
                                                                        elif ENCODER == 'SDCSE':
                                                                            result_path = os.path.join(root_path, f'{mode}_{taskset}_unsup_{ENCODER.lower()}_{plm}_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_w}_{pt_type}_{pt_num}_{pt_step}_{loss}_{pooler}_{metric}_{margin}.txt')
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
                                                                        result = [mode, taskset, plm, bs, lr, epoch, max_len, lambda_w, pt_type, pt_num, pt_step, loss, pooler, metric, margin, seed] + list(map(float, list_score))
                                                                        
                                                                        dict_result[result_path] = result
                                                                    except:
                                                                        pass

    df = pd.DataFrame(dict_result.values(), columns=['mode', 'taskset', 'plm', 'bs', 'lr', 'epoch', 'max_len', 'lambda_w', 'pt_type', 'pt_num', 'pt_step', 'loss', 'pooler', 'metric', 'margin', 'seed'] + list_sts + list_transfer + ['sts', 'transfer'])
    df_groupby = df.groupby([*groupby]).agg([np.mean, np.std])
    # if df_groupby.index.name == 'lambda_w':
    #     # print(df_groupby.loc[list_lambda_w])
    return df, df_groupby

ENCODER = 'SimCSE'
RESULT_FOLDER = 'backup'

ENCODER = 'SDCSE'
RESULT_FOLDER = 'backup_eval_dropout_sim1_all'
RESULT_FOLDER = 'backup_eval_dropout_sim1_nocls'
RESULT_FOLDER = 'backup_eval_dropout_sim0_nocls'
# RESULT_FOLDER = 'backup_eval_dropout_sim0_all'

result_dev('plm', 'bs', 'loss', 'lambda_w', 'pt_type', 'pooler', 'pt_num', 'pt_step', 'margin')[1]
result_eval('mode', 'taskset', 'plm', 'bs', 'loss', 'lambda_w', 'pt_type', 'pooler', 'pt_num', 'pt_step', 'margin')[1].loc[:, ['sts', 'transfer']].round(1)


x = result_eval('mode', 'taskset', 'plm', 'bs', 'loss', 'lambda_w', 'pt_type', 'pooler', 'pt_num', 'pt_step', 'margin')[1].loc[:, list_sts + ['sts']]
x[pd.MultiIndex.from_product([list_sts + ['sts'], ['std']])] = x[pd.MultiIndex.from_product([list_sts + ['sts'], ['std']])] * 1.96 / np.sqrt(5)
x = x.round(2).apply(lambda x: [format(xi, ".1f") for xi in x])
for col in list_sts + ['sts']:
    x.loc[:, (col, 'mean')] = r'{' + x.loc[:, (col, 'mean')] + r'}$_{\pm' + x.loc[:, (col, 'std')] + '}$ &'
x[pd.MultiIndex.from_product([list_sts + ['sts'], ['mean']])]


x = result_eval('mode', 'taskset', 'plm', 'bs', 'loss', 'lambda_w', 'pt_type', 'pooler', 'pt_num', 'pt_step', 'margin')[1].loc[:, list_transfer + ['transfer']]
x[pd.MultiIndex.from_product([list_transfer + ['transfer'], ['std']])] = x[pd.MultiIndex.from_product([list_transfer + ['transfer'], ['std']])] * 1.96 / np.sqrt(5)
x = x.round(2).apply(lambda x: [format(xi, ".1f") for xi in x])
for col in list_transfer + ['transfer']:
    x.loc[:, (col, 'mean')] = r'{' + x.loc[:, (col, 'mean')] + r'}$_{\pm' + x.loc[:, (col, 'std')] + '}$ &'
x[pd.MultiIndex.from_product([list_transfer + ['transfer'], ['mean']])]
