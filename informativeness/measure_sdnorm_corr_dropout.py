from re import A
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import os
from itertools import chain, product
import pandas as pd
from torch.nn.parallel import DataParallel
import json

PATH_DATA = os.path.join(os.getcwd(), 'data', 'wiki1m_for_simcse.txt')
with open(PATH_DATA, 'r') as f:
    list_text = f.readlines()
list_text = np.random.choice(list_text, size=100000, replace=False)
len(list_text)

# FILE_NAME = 'model_meta_data_open.json'
FILE_NAME = 'model_meta_data_my.json'
with open(FILE_NAME, 'r') as f:
    dict_model = json.load(f)
    list_encoder_to_remove = ['diffcse', 'promcse', 'scd']
    # list_encoder_to_remove = ['promcse', 'scd']
    for encoder in list_encoder_to_remove:
        dict_model.pop(encoder) 

INIT_DROPOUT = 0
BATCH_SIZE = 64
DEVICE = 'cuda'
MAX_LEN = 32
SENTENCE_PAIR_SIZE = 6
assert SENTENCE_PAIR_SIZE <= 10
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

list_perturb_num = range(1, 2)
list_perturb_step = np.arange(0.1, round(0.1 * SENTENCE_PAIR_SIZE, 1), 0.1).round(2)
list_product = list(product(list_perturb_num, list_perturb_step))
# list_init_dropout = [0, 0.1]
list_init_dropout = [0]

columns = pd.MultiIndex.from_product([dict_model.keys(), list(dict_model.values())[0].keys(), list(list(dict_model.values())[0].values())[0].keys(), list_init_dropout], names=['encoder', 'plm', 'size', 'init_dropout'])
index = pd.MultiIndex.from_tuples(list_product, names=['perturb_num', 'perturb_dropout'])
df = pd.DataFrame(index=index, columns=columns)
# encoder, plm, size, perturb_num = 'simcse', 'roberta', 'large', 1
for encoder in tqdm(dict_model):
    for plm in tqdm(dict_model[encoder], leave=False):
        for size in tqdm(dict_model[encoder][plm], leave=False):
            model_path = dict_model[encoder][plm][size]
            for perturb_num in list_perturb_num:
                model = AutoModel.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                class CustomDropout(nn.Module):
                    def __init__(self, dict_dropout):
                        super(CustomDropout, self).__init__()
                        self.dict_dropout = dict_dropout
                        self.dropout = nn.Dropout(0.1)
                        for i in range(len(dict_dropout)):
                            setattr(self, f'dropout_{i}', nn.Dropout(dict_dropout[f'dropout_{i}']))
                        
                    def forward(self, x):
                        out = x.clone()
                        for i in range(len(self.dict_dropout)):
                            dropout_i = getattr(self, f'dropout_{i}')
                            # out[i::len(self.dict_dropout), :, :] = dropout_i(x[i::len(self.dict_dropout), :, :])
                            # p = dropout_i.p #if i else 0
                            out[i::SENTENCE_PAIR_SIZE, 1:, :] = dropout_i(x[i::SENTENCE_PAIR_SIZE, 1:, :]) #* (1-p) / 0.9
                        out[:, 0, :] = self.dropout_0(out[:, 0, :])
                        return out
                
                # BERT 모델에서 드롭아웃 레이어 찾기
                dropout_layer_names = []
                for name, module in model.named_modules():
                    if isinstance(module, nn.Dropout) or isinstance(module, CustomDropout):
                        dropout_layer_names.append(name)
                # dropout_layer_names

                # 드롭아웃 레이어 변경
                dict_dropout = {f'dropout_{i}': round(INIT_DROPOUT + 0.1 * i, 4) for i in range(SENTENCE_PAIR_SIZE)}
                custom_dropout = CustomDropout(dict_dropout)
                for name in dropout_layer_names:
                    if name.startswith('embeddings'):
                        model.embeddings.dropout = custom_dropout
                    elif name.startswith('encoder'):
                        n = int(name.split('.')[2])
                        model.encoder.layer[n].attention.self.dropout = custom_dropout
                        model.encoder.layer[n].attention.output.dropout = custom_dropout
                        model.encoder.layer[n].output.dropout = custom_dropout
                # model
                
                if torch.cuda.device_count() > 1:
                    model = DataParallel(model)
                    batch_size = BATCH_SIZE * torch.cuda.device_count()
                else:
                    batch_size = int(BATCH_SIZE // SENTENCE_PAIR_SIZE)
                _ = model.to(DEVICE)
                
                list_batch = [list_text[i:i+batch_size] for i in range(0, len(list_text), batch_size)]
                len(list_batch)

                def func(batch):
                    batch_augmented = list(chain.from_iterable(zip(*[batch] * (SENTENCE_PAIR_SIZE))))
                    tokenized_input = tokenizer(batch_augmented, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                    tokenized_input = {k: v.to(DEVICE) for k, v in tokenized_input.items()}
                    output = model(**tokenized_input, return_dict=True)
                    output = output.last_hidden_state.cpu().detach()
                    return output

                list_corr = [[[] for _ in range(len(list_init_dropout))] for i in range(SENTENCE_PAIR_SIZE - 1)]
                for batch in tqdm(list_batch, leave=False):
                    output = func(batch)
                    norm = output[:, 0].norm(dim=-1)
                    norm_reshape = norm.reshape(-1, SENTENCE_PAIR_SIZE)
                    for i in range(len(list_init_dropout)): # type of intact dropout ratio
                        for j in range(1, SENTENCE_PAIR_SIZE):
                            if i == j:
                                list_corr[j - 1][i].append(np.nan)
                                continue
                            norm_selected = norm_reshape[:, [i, j]]
                            x = torch.argsort(norm_selected, descending=True)
                            y = torch.Tensor([range(2) for i in range(len(x))])
                            corr_temp = torch.mean(1 - (x - y).pow(2).sum(dim=1).mul(6).div((perturb_num + 1) * ((perturb_num + 1) ** 2 - 1)))
                            list_corr[j - 1][i].append(corr_temp.item())
                    # for i in range(1, SENTENCE_PAIR_SIZE):
                    #     norm_selected = norm_reshape[:, [0, i]]
                    #     x = torch.argsort(norm_selected, descending=True)
                    #     y = torch.Tensor([range(2) for i in range(batch_size)])
                    #     corr_temp = torch.mean(1 - (x - y).pow(2).sum(dim=1).mul(6).div((perturb_num + 1) * ((perturb_num + 1) ** 2 - 1)))
                    #     list_corr[i - 1].append(corr_temp.item())
                list_corr_mean = np.array(list_corr).mean(axis=2)
                df.loc[perturb_num, (encoder, plm, size)] = list_corr_mean
df = df.round(2)
df.to_csv(f"../sdnorm_dropout_{FILE_NAME.split('.')[0].split('_')[-1]}.csv")