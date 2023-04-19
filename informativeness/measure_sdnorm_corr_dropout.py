from re import A
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import os
from itertools import chain, product
import pandas as pd

PATH_DATA = os.path.join(os.getcwd(), 'data', 'wiki1m_for_simcse.txt')
with open(PATH_DATA, 'r') as f:
    list_text = f.readlines()
len(list_text)

# BERT 모델 불러오기
dict_model_name = {
    'bert-base': 'princeton-nlp/unsup-simcse-bert-base-uncased',
    'bert-large': 'princeton-nlp/unsup-simcse-bert-large-uncased',
    'roberta-base': 'princeton-nlp/unsup-simcse-roberta-base',
    'roberta-large': 'princeton-nlp/unsup-simcse-roberta-large',
}
list_perturb_num = range(1, 10)
list_perturb_step = np.arange(1, 10)
list_product = list(product(list_perturb_num, list_perturb_step))

df = pd.DataFrame(index=pd.MultiIndex.from_tuples(list_product, names=['list_perturb_num', 'list_perturb_step']), columns=dict_model_name)
for model_name_simple, model_name_full in dict_model_name.items():
    for perturb_num in list_perturb_num:
        for perturb_step in list_perturb_step:
            if 0.1 + 0.1 * perturb_step * perturb_num > 1:
                continue
            model = AutoModel.from_pretrained(model_name_full)
            tokenizer = AutoTokenizer.from_pretrained(model_name_full)

            # 드롭아웃 확률 정의
            init_dropout = 0.1
            batch_size = int(128 // perturb_num)
            max_length = 32
            device='cuda:3'
            dict_dropout = {f'dropout_{i}': round(init_dropout + 0.1 * perturb_step * i, 4) for i in range(perturb_num + 1)}

            class CustomDropout(nn.Module):
                def __init__(self, dict_dropout, perturb_num):
                    super(CustomDropout, self).__init__()
                    self.perturb_num = perturb_num
                    for i in range(perturb_num + 1):
                        setattr(self, f'dropout_{i}', nn.Dropout(dict_dropout[f'dropout_{i}']))
                    
                def forward(self, x):
                    out = x.clone()
                    for i in range(self.perturb_num + 1):
                        out[i::self.perturb_num + 1, 1:, :] = getattr(self, f'dropout_{i}')(x[i::self.perturb_num + 1, 1:, :])
                    return out

            # BERT 모델에서 드롭아웃 레이어 찾기
            dropout_layer_names = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Dropout) or isinstance(module, CustomDropout):
                    dropout_layer_names.append(name)
            dropout_layer_names

            # 드롭아웃 레이어 변경
            dict_dropout = {f'dropout_{i}': round(0.1 + 0.1 * perturb_step * i, 4) for i in range(perturb_num + 1)}
            custom_dropout = CustomDropout(dict_dropout, perturb_num)
            for name in dropout_layer_names:
                if name.startswith('embeddings'):
                    model.embeddings.dropout = custom_dropout
                elif name.startswith('encoder'):
                    n = int(name.split('.')[2])
                    model.encoder.layer[n].attention.self.dropout = custom_dropout
                    model.encoder.layer[n].attention.output.dropout = custom_dropout
                    model.encoder.layer[n].output.dropout = custom_dropout
            model

            _ = model.to(device)

            list_batch = [list_text[i:i+batch_size] for i in range(0, len(list_text), batch_size)]

            # 모델에 입력 텐서 전달
            def func(batch):
                batch = list(chain.from_iterable(zip(*[batch] * (perturb_num + 1))))
                tokenized_input = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
                output = model(**tokenized_input, return_dict=True)
                output = output.last_hidden_state.cpu().detach()
                return output

            # x = model.embeddings.word_embeddings(tokenized_input['input_ids'])
            # y = model.embeddings.dropout(x)
            # sum(y[0, 0, :] == 0)
            # sum(y[1, 0, :] == 0)

            list_corr = []
            for batch in tqdm(list_batch[:5]):
                output = func(batch)
                norm = output[:, 0].norm(dim=-1)
                norm_reshape = norm.reshape(batch_size, perturb_num + 1)
                x = norm_reshape.argsort()
                y = torch.Tensor([range(perturb_num, -1, -1) for i in range(batch_size)])
                corr_temp = torch.mean(1 - (x - y).pow(2).sum(dim=1).mul(6).div((perturb_num + 1) * ((perturb_num + 1) ** 2 - 1)))
                list_corr.append(corr_temp)
            corr = np.array(list_corr).mean()
            df.loc[(perturb_num, perturb_step), model_name_simple] = corr
df = df.dropna(axis=0, how='all')
df['mean'] = df.mean(axis=1)
df.to_csv('../sdnorm_dropout.csv')