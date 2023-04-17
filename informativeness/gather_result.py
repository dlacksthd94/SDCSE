import os
import pandas as pd
import re
from itertools import chain

pd.set_option('display.max_rows', 100)

def result_dev(*groupby):
    root_path = os.path.join(os.path.expanduser('~'), 'PAPER/SDCSE/informativeness/', ENCODER, 'result', RESULT_FOLDER)
    assert os.path.exists(root_path)

    list_result = []
    for batch_size in [128]:
        for lr in [f'1e-{i}' for i in range(4, 5)]:
            for epoch in range(1, 2):
                for max_len in [32]:
                    list_lambda_weight = ['0e-0'] + [f'1e-{i}' for i in range(0, 1)]
                    for lambda_weight in list_lambda_weight:
                        for perturb_type in ['mask_token', 'unk_token', 'pad_token']:
                            for perturb_num in range(1, 4):
                                for perturb_step in range(1, 4):
                                    for seed in range(0, 1):
                                        for loss_type in ['l1', 'sl1', 'mse']:
                                            for pooler in ['wp', 'wop']:
                                                for metric in ['stsb', 'sickr', 'sts', 'transfer']:
                                                    try:
                                                        result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-bert-base-uncased_{batch_size}_{lr}_{epoch}_{seed}_{max_len}_{lambda_weight}_{perturb_type}_{perturb_num}_{perturb_step}_{loss_type}_{pooler}_{metric}', 'eval_results.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-bert-base-uncased_{batch_size}_{lr}_{epoch}_{seed}_{max_len}_{lambda_weight}', 'eval_results.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-bert-base-uncased_{batch_size}_{lr}_{epoch}_{seed}_{max_len}', 'eval_results.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-bert-base-uncased_{batch_size}_{lr}_{epoch}_{seed}', 'eval_results.txt')
                                                        assert os.path.exists(result_path)
                                                        df_temp = pd.read_csv(result_path, sep='=', header=None)
                                                        df_temp = df_temp[-4:-2]
                                                        df_temp.columns = ['task', 'score']
                                                        result = [batch_size, lr, epoch, max_len, lambda_weight, perturb_type, perturb_num, perturb_step, loss_type, pooler, metric, seed] + df_temp['score'].to_list()
                                                        list_result.append(result)
                                                    except:
                                                        os.path.join(root_path, f'my-unsup-{ENCODER.lower()}-bert-base-uncased_{batch_size}_{lr}_{epoch}_{seed}_{max_len}_{lambda_weight}_{perturb_type}_{perturb_num}_{perturb_step}_{loss_type}_{pooler}', 'eval_results.txt')
                                                        pass
    df = pd.DataFrame(list_result, columns=['batch_size', 'lr', 'epoch', 'max_len', 'lambda_weight', 'perturb_type', 'perturb_num', 'perturb_step', 'loss_type', 'pooler', 'metric', 'seed', 'sts', 'transfer'])
    df_groupby = df.groupby([*groupby])['sts', 'transfer'].agg(['mean', 'std'])
    # if df_groupby.index.name == 'lambda_weight':
    #     # print(df_groupby.loc[list_lambda_weight])
    return df, df_groupby

def result_eval(*groupby):
    root_path = os.path.join(os.path.expanduser('~'), 'PAPER/SDCSE/informativeness/result/evaluation', ENCODER.lower(), RESULT_FOLDER)
    assert os.path.exists(root_path)

    list_result = []
    for batch_size in [128]:
        for lr in [f'1e-{i}' for i in range(4, 5)]:
            for epoch in range(1, 2):
                for max_len in [32]:
                    list_lambda_weight = ['0e-0'] + [f'1e-{i}' for i in range(0, 1)]
                    for lambda_weight in list_lambda_weight:
                        for perturb_type in ['mask_token', 'unk_token', 'pad_token']:
                            for perturb_num in range(1, 4):
                                for perturb_step in range(1, 4):
                                    for seed in range(0, 1):
                                        for loss_type in ['l1', 'sl1', 'mse']:
                                            for pooler in ['wp', 'wop']:
                                                for metric in ['stsb', 'sickr', 'sts', 'transfer']:
                                                    try:
                                                        result_path = os.path.join(root_path, f'result_unsup_{ENCODER.lower()}_bert_{batch_size}_{lr}_{epoch}_{seed}_{max_len}_{lambda_weight}_{perturb_type}_{perturb_num}_{perturb_step}_{loss_type}_{pooler}_{metric}.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'result_unsup_{ENCODER.lower()}_bert_{batch_size}_{lr}_{epoch}_{seed}_{max_len}_{lambda_weight}.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'result_unsup_{ENCODER.lower()}_bert_{batch_size}_{lr}_{epoch}_{seed}_{max_len}.txt')
                                                        if not os.path.exists(result_path):
                                                            result_path = os.path.join(root_path, f'result_unsup_{ENCODER.lower()}_bert_{batch_size}_{lr}_{epoch}_{seed}.txt')
                                                        assert os.path.exists(result_path)
                                                        
                                                        with open(result_path, 'r') as f:
                                                            text = f.read()
                                                        text = re.sub(r'-* fasttest -*', '', text).strip()
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
                                                        df2 = prettytable_to_dataframe(text2)
                                                        list_score = [df1['Avg.'].squeeze(), df2['Avg.'].squeeze()]
                                                        result = [batch_size, lr, epoch, max_len, lambda_weight, perturb_type, perturb_num, perturb_step, loss_type, pooler, metric, seed] + list(map(float, list_score))
                                                        
                                                        list_result.append(result)
                                                    except:
                                                        pass
    df = pd.DataFrame(list_result, columns=['batch_size', 'lr', 'epoch', 'max_len', 'lambda_weight', 'perturb_type', 'perturb_num', 'perturb_step', 'loss_type', 'pooler', 'metric', 'seed', 'sts', 'transfer'])
    df_groupby = df.groupby([*groupby])['sts', 'transfer'].agg(['mean', 'std'])
    # if df_groupby.index.name == 'lambda_weight':
    #     # print(df_groupby.loc[list_lambda_weight])
    return df, df_groupby

ENCODER = 'SDCSE'
RESULT_FOLDER = 'backup_eval_token_sim1'

result_dev('loss_type', 'perturb_type', 'perturb_num', 'perturb_step')[1]
result_eval('perturb_type', 'pooler')[1]