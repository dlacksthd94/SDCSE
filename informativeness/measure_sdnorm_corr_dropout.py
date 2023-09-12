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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

PATH_DATA = os.path.join(os.getcwd(), 'data', 'wiki1m_for_simcse.txt')
SAMPLE_SIZE = 1000
with open(PATH_DATA, 'r') as f:
    list_text = f.readlines()
list_text = np.random.choice(list_text, size=SAMPLE_SIZE, replace=False)
len(list_text)

dict_meta_data = {
    'BERT': 'bert-base-uncased',
    'SimCSE': 'SDCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-sdcse-bert-base-uncased_64_3e-5_1_0_32_0e-0_none_0_0_mse_wp_stsb_0e-0_0e-0_0e-0_0',
    'SimCSE+': 'SDCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-sdcse-bert-base-uncased_64_3e-5_1_0_32_1e-2_dropout_1_2_margin_wp_stsb_1e-1_0e-0_0e-0_0',
    'DiffCSE': 'DiffCSE/result/backup_eval_dropout_sim0_nocls_sts_1gpu/my-unsup-diffcse-bert-base-uncased_64_7e-6_2_0_32_0e-0_none_0_0_mse_wp_stsb_0e-0_5e-3_3e-1_0',
    'DiffCSE+': 'DiffCSE/result/backup_eval_dropout_sim0_nocls_sts_1gpu/my-unsup-diffcse-bert-base-uncased_64_7e-6_2_0_32_1e-2_dropout_1_2_margin_wp_stsb_1e-2_5e-3_3e-1_0',
    'PromCSE': 'PromCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-promcse-bert-base-uncased_256_3e-2_1_0_32_0e-0_none_0_0_mse_wp_stsb_0e-0_0e-0_0e-0_16',
    # 'PromCSE+': 'PromCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-promcse-bert-base-uncased_256_3e-2_1_0_32_1e-1_dropout_1_2_margin_wp_stsb_1e-1_0e-0_0e-0_16',
    'PromCSE+': 'PromCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-promcse-bert-base-uncased_256_3e-2_1_0_32_5e-0_dropout_1_2_margin_wp_stsb_1e-1_0e-0_0e-0_16',
    'MixCSE': 'MixCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-mixcse-bert-base-uncased_64_3e-5_1_0_32_0e-0_none_0_0_mse_wp_stsb_0e-0_0e-0_0e-0_0',
    'MixCSE+': 'MixCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-mixcse-bert-base-uncased_64_3e-5_1_0_32_1e-2_dropout_1_2_margin_wp_stsb_1e-1_0e-0_0e-0_0',
}

INIT_DROPOUT = 0.0
DROPOUT_STEP = 0.2
INFORMATIVE_PAIR_SIZE = 2
BATCH_SIZE = 64
DEVICE = 'cuda'
MAX_LEN = 32

dict_meta_data

df = pd.DataFrame(index=['corr'], columns=dict_meta_data.keys())
for model_name in dict_meta_data.keys():
    model_path = dict_meta_data[model_name]
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
                out[i::INFORMATIVE_PAIR_SIZE, 1:, :] = dropout_i(x[i::INFORMATIVE_PAIR_SIZE, 1:, :]) #* (1-p) / 0.9
            out[:, 0, :] = self.dropout(out[:, 0, :])
            return out

    # BERT 모델에서 드롭아웃 레이어 찾기
    dropout_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) or isinstance(module, CustomDropout):
            dropout_layer_names.append(name)
    # dropout_layer_names

    # 드롭아웃 레이어 변경
    dict_dropout = {f'dropout_{i}': round(INIT_DROPOUT + DROPOUT_STEP * i, 1) for i in range(INFORMATIVE_PAIR_SIZE)}
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
        batch_size = int(BATCH_SIZE // INFORMATIVE_PAIR_SIZE)
    _ = model.to(DEVICE)

    list_batch = [list_text[i:i+batch_size] for i in range(0, len(list_text), batch_size)]

    def func(batch):
        batch_augmented = list(chain.from_iterable(zip(*[batch] * (INFORMATIVE_PAIR_SIZE))))
        tokenized_input = tokenizer(batch_augmented, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        tokenized_input = {k: v.to(DEVICE) for k, v in tokenized_input.items()}
        output = model(**tokenized_input, return_dict=True)
        output = output.last_hidden_state.cpu().detach()
        return output

    list_corr = []
    for batch in tqdm(list_batch, leave=False):
        output = func(batch)
        norm = output[:, 0].norm(dim=-1)
        norm_reshape = norm.reshape(-1, INFORMATIVE_PAIR_SIZE)
        list_corr.append(norm_reshape)
    norm_total = torch.cat(list_corr)
    x = torch.argsort(norm_total, descending=True)
    y = torch.Tensor([range(INFORMATIVE_PAIR_SIZE) for i in range(len(x))])
    corr_temp = torch.mean(1 - (x - y).pow(2).sum(dim=1).mul(6).div((INFORMATIVE_PAIR_SIZE + 1) * ((INFORMATIVE_PAIR_SIZE + 1) ** 2 - 1)))
    df.loc['corr', model_name] = corr_temp.item()

df = df.round(2)
df
# df.to_csv(f"../sdnorm_dropout_{FILE_NAME.split('.')[0].split('_')[-1]}.csv")