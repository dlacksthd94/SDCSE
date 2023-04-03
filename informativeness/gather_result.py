import os
import pandas as pd
import re

ENCODER = 'SimCSE'
RESULT_FOLDER = 'backup_no_mlm_diff_seed_eval_transfer'

def result_dev(*groupby):
    root_path = os.path.join(os.path.expanduser('~'), 'NLP/PROJECT/GSDS_NLP_sentence-embedding/informativeness/', ENCODER, 'result', RESULT_FOLDER)
    assert os.path.exists(root_path)

    list_result = []
    for batch_size in [64, 128, 256]:
        for lr in [f'1e-{i}' for i in range(4, 5)]:
            for epoch in range(1, 2):
                for seed in range(0, 10):
                    try:
                        result_path = os.path.join(root_path, f'my-unsup-simcse-bert-base-uncased_{batch_size}_{lr}_{epoch}_{seed}', 'eval_results.txt')
                        assert os.path.exists(result_path)
                        df_temp = pd.read_csv(result_path, sep='=', header=None)
                        df_temp = df_temp[-4:-2]
                        df_temp.columns = ['task', 'score']
                        result = [batch_size, lr, epoch, seed] + df_temp['score'].to_list()
                        list_result.append(result)
                    except:
                        pass
    df = pd.DataFrame(list_result, columns=['batch_size', 'lr', 'epoch', 'seed', 'sts', 'transfer'])
    print(df.groupby([*groupby])['sts', 'transfer'].agg(['mean', 'std']))
    
def result_eval(*groupby):
    root_path = os.path.join(os.path.expanduser('~'), 'NLP/PROJECT/GSDS_NLP_sentence-embedding/informativeness/result/evaluation', ENCODER.lower(), RESULT_FOLDER)
    assert os.path.exists(root_path)

    list_result = []
    for batch_size in [64, 128, 256]:
        for lr in [f'1e-{i}' for i in range(4, 5)]:
            for epoch in range(1, 2):
                for seed in range(0, 10):
                    try:
                        result_path = os.path.join(root_path, f'result_unsup_{ENCODER.lower()}_bert_wop_{batch_size}_{lr}_{epoch}_{seed}.txt')
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
                        result = [batch_size, lr, epoch, seed] + list(map(float, list_score))
                        
                        list_result.append(result)
                    except:
                        pass
    df = pd.DataFrame(list_result, columns=['batch_size', 'lr', 'epoch', 'seed', 'sts', 'transfer'])
    print(df.groupby([*groupby])['sts', 'transfer'].agg(['mean', 'std']))

result_dev('batch_size')
result_eval('batch_size')