import os
import pandas as pd
import re
from itertools import chain

pd.set_option('display.max_rows', 100)

def result_dev(*groupby):
    root_path = os.path.join(os.path.expanduser('~'), 'PAPER/SDCSE/informativeness/', ENCODER, 'result', RESULT_FOLDER)
    assert os.path.exists(root_path)

    list_result = []
    for bs in [128]:
        for lr in [f'1e-{i}' for i in range(4, 5)]:
            for epoch in range(1, 2):
                for max_len in [32]:
                    list_lambda_w = ['0e-0'] + [f'1e-{i}' for i in range(0, 2)]
                    for lambda_w in list_lambda_w:
                        for pt_type in ['mask_token', 'unk_token', 'pad_token', 'dropout']:
                            for pt_num in range(1, 4):
                                for pt_step in range(1, 4):
                                    for seed in range(0, 1):
                                        for loss in ['l1', 'sl1', 'mse']:
                                            for pooler in ['wp', 'wop']:
                                                for metric in ['stsb', 'sickr', 'sts', 'transfer']:
                                                    try:
                                                        result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-bert-base-uncased_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_w}_{pt_type}_{pt_num}_{pt_step}_{loss}_{pooler}_{metric}', 'eval_results.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-bert-base-uncased_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_w}', 'eval_results.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-bert-base-uncased_{bs}_{lr}_{epoch}_{seed}_{max_len}', 'eval_results.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-bert-base-uncased_{bs}_{lr}_{epoch}_{seed}', 'eval_results.txt')
                                                        assert os.path.exists(result_path)
                                                        df_temp = pd.read_csv(result_path, sep='=', header=None)
                                                        df_temp = df_temp[-4:-2].reset_index(drop=True)
                                                        df_temp.columns = ['task', 'score']
                                                        result = [bs, lr, epoch, max_len, lambda_w, pt_type, pt_num, pt_step, loss, pooler, metric, seed, round(df_temp['score'][0] * 100, 2), round(df_temp['score'][1], 2)]
                                                        list_result.append(result)
                                                    except:
                                                        os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-bert-base-uncased_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_w}_{pt_type}_{pt_num}_{pt_step}_{loss}_{pooler}', 'eval_results.txt')
                                                        pass
    df = pd.DataFrame(list_result, columns=['bs', 'lr', 'epoch', 'max_len', 'lambda_w', 'pt_type', 'pt_num', 'pt_step', 'loss', 'pooler', 'metric', 'seed', 'sts', 'transfer'])
    df_groupby = df.groupby([*groupby])['sts', 'transfer'].agg(['mean', 'std'])
    # if df_groupby.index.name == 'lambda_w':
    #     # print(df_groupby.loc[list_lambda_w])
    return df, df_groupby

def result_eval(*groupby):
    root_path = os.path.join(os.path.expanduser('~'), 'PAPER/SDCSE/informativeness/result/evaluation', ENCODER.lower(), RESULT_FOLDER)
    assert os.path.exists(root_path)

    list_result = []
    for bs in [128]:
        for lr in [f'1e-{i}' for i in range(4, 5)]:
            for epoch in range(1, 2):
                for max_len in [32]:
                    list_lambda_w = ['0e-0'] + [f'1e-{i}' for i in range(0, 2)]
                    for lambda_w in list_lambda_w:
                        for pt_type in ['mask_token', 'unk_token', 'pad_token', 'dropout']:
                            for pt_num in range(1, 4):
                                for pt_step in range(1, 4):
                                    for seed in range(0, 1):
                                        for loss in ['l1', 'sl1', 'mse']:
                                            for pooler in ['wp', 'wop']:
                                                for metric in ['stsb', 'sickr', 'sts', 'transfer']:
                                                    try:
                                                        result_path = os.path.join(root_path, f'result_unsup_{ENCODER.lower()}_bert_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_w}_{pt_type}_{pt_num}_{pt_step}_{loss}_{pooler}_{metric}.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'result_unsup_{ENCODER.lower()}_bert_{bs}_{lr}_{epoch}_{seed}_{max_len}_{lambda_w}.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'result_unsup_{ENCODER.lower()}_bert_{bs}_{lr}_{epoch}_{seed}_{max_len}.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'result_unsup_{ENCODER.lower()}_bert_{bs}_{lr}_{epoch}_{seed}.txt')
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
                                                        if text2:
                                                            df2 = prettytable_to_dataframe(text2)
                                                            list_score = [df1['Avg.'].squeeze(), df2['Avg.'].squeeze()]
                                                        else:
                                                            list_score = [df1['Avg.'].squeeze(), 0]
                                                        result = [bs, lr, epoch, max_len, lambda_w, pt_type, pt_num, pt_step, loss, pooler, metric, seed] + list(map(float, list_score))
                                                        
                                                        list_result.append(result)
                                                    except:
                                                        pass
    df = pd.DataFrame(list_result, columns=['bs', 'lr', 'epoch', 'max_len', 'lambda_w', 'pt_type', 'pt_num', 'pt_step', 'loss', 'pooler', 'metric', 'seed', 'sts', 'transfer'])
    df_groupby = df.groupby([*groupby])['sts', 'transfer'].agg(['mean', 'std'])
    # if df_groupby.index.name == 'lambda_w':
    #     # print(df_groupby.loc[list_lambda_w])
    return df, df_groupby

ENCODER = 'SimCSE'
RESULT_FOLDER = 'backup'

ENCODER = 'SDCSE'
RESULT_FOLDER = 'backup_eval_dropout_sim1_all'
RESULT_FOLDER = 'backup_eval_dropout_sim1_nocls'
RESULT_FOLDER = 'backup_eval_token_sim1'
RESULT_FOLDER = '../../../../../../../../../../data/chansonglim/backup_eval_token_sim1'

result_dev('loss', 'pt_type', 'pt_num', 'pt_step')[0]
result_eval('loss', 'pt_type', 'pt_num', 'pt_step', 'pooler')[1]