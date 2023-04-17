import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, AutoModel
from scipy.stats import spearmanr
from tqdm import tqdm
import numpy as np

# BERT 모델 불러오기
model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased')

# 드롭아웃 확률 정의
init_dropout = 0.1
perturb_num = 2
perturb_step = 1.5
batch_size = 128
dict_dropout = {f'dropout_{i}': round(init_dropout + 0.1 * perturb_step * i, 4) for i in range(perturb_num + 1)}

# 사용자 정의 드롭아웃 레이어
class CustomDropout(nn.Module):
    def __init__(self, dict_dropout):
        super(CustomDropout, self).__init__()
        self.list_dropout = []
        for i in range(perturb_num + 1):
            setattr(self, f'dropout_{i}', nn.Dropout(dict_dropout[f'dropout_{i}']))
            # self.list_dropout[i] = nn.Dropout(dict_dropout[f'dropout_{i}'])
        
    def forward(self, x):
        out = torch.zeros_like(x)
        for i in range(perturb_num + 1):
            out[i::perturb_num + 1] = getattr(self, f'dropout_{i}')(x[i::perturb_num + 1])
            # out[i::perturb_num + 1] = self.list_dropout[i](x[i::perturb_num + 1])
        return out

# BERT 모델에서 드롭아웃 레이어 찾기
dropout_layer_names = []
for name, module in model.named_modules():
    if isinstance(module, nn.Dropout):
        dropout_layer_names.append(name)
dropout_layer_names

# 드롭아웃 레이어 변경
custom_dropout = CustomDropout(dict_dropout)
for name in dropout_layer_names:
    if name.startswith('embeddings'):
        model.embeddings.dropout = custom_dropout
    elif name.startswith('encoder'):
        n = int(name.split('.')[2])
        model.encoder.layer[n].attention.self.dropout = custom_dropout
        model.encoder.layer[n].attention.output.dropout = custom_dropout
        model.encoder.layer[n].output.dropout = custom_dropout
model

# 입력 텐서
text = 'Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as noise.'
text = 'The sound of laughter echoed through the room, filling me with a sense of joy and happiness.'
# text = 'I love listening to music while I work.'
device='cuda'
list_text = [text] * (perturb_num + 1) * batch_size
tokenized_inputs = tokenizer(list_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
_ = model.to(device)

# 모델에 입력 텐서 전달
def func(tokenized_inputs):
    output = model(**tokenized_inputs)
    return output[0].cpu().detach()

list_corr = []
for i in tqdm(range(30)):
    output = func(tokenized_inputs)
    norm = output[:, 0].norm(dim=-1)
    norm_reshape = norm.reshape(batch_size, perturb_num + 1)
    x = norm_reshape.argsort()
    y = torch.Tensor([range(perturb_num, -1, -1) for i in range(batch_size)])
    corr_temp = torch.mean(1 - (x - y).pow(2).sum(dim=1).mul(6).div((perturb_num + 1) * ((perturb_num + 1) ** 2 - 1)))
    list_corr.append(corr_temp)
np.array(list_corr).mean()