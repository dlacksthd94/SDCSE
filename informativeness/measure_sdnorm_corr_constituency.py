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
import pickle
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

PATH_DATA = os.path.join(os.getcwd(), 'data', 'backup_100000', 'wiki1m_tree_cst_lg_large_subsentence.pickle')
SAMPLE_SIZE = 10000
with open(PATH_DATA, 'rb') as f:
    list_text = pickle.load(f)
list_text = np.random.choice(list_text, size=SAMPLE_SIZE, replace=False)
len(list_text)

dict_meta_data = {
    'BERT': 'bert-base-uncased',
    'SimCSE': 'SDCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-sdcse-bert-base-uncased_64_3e-5_1_0_32_0e-0_none_0_0_mse_wp_stsb_0e-0_0e-0_0e-0_0',
    'SimCSE+': 'SDCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-sdcse-bert-base-uncased_64_3e-5_1_0_32_1e-2_dropout_1_2_margin_wp_stsb_1e-1_0e-0_0e-0_0',
    'DiffCSE': 'DiffCSE/result/backup_eval_dropout_sim0_nocls_sts_1gpu/my-unsup-diffcse-bert-base-uncased_64_7e-6_2_0_32_0e-0_none_0_0_mse_wp_stsb_0e-0_5e-3_3e-1_0',
    'DiffCSE+': 'DiffCSE/result/backup_eval_dropout_sim0_nocls_sts_1gpu/my-unsup-diffcse-bert-base-uncased_64_7e-6_2_0_32_1e-2_dropout_1_2_margin_wp_stsb_1e-2_5e-3_3e-1_0',
    'PromCSE': 'PromCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-promcse-bert-base-uncased_256_3e-2_1_0_32_0e-0_none_0_0_mse_wp_stsb_0e-0_0e-0_0e-0_16',
    'PromCSE+': 'PromCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-promcse-bert-base-uncased_256_3e-2_1_0_32_1e-1_dropout_1_2_margin_wp_stsb_1e-1_0e-0_0e-0_16',
    'MixCSE': 'MixCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-mixcse-bert-base-uncased_64_3e-5_1_1_32_0e-0_none_0_0_mse_wp_stsb_0e-0_0e-0_0e-0_0',
    'MixCSE+': 'MixCSE/result/backup_eval_dropout_sim0_nocls_1gpu/my-unsup-mixcse-bert-base-uncased_64_3e-5_1_1_32_1e-2_dropout_1_2_margin_wp_stsb_1e-1_0e-0_0e-0_0',
}

PERTURB_STEP = 1
INFORMATIVE_PAIR_SIZE = 5
BATCH_SIZE = 64
DEVICE = 'cuda'
MAX_LEN = 32

# make informative pair
list_text = list(filter(lambda x: len(x) >= 1 + (INFORMATIVE_PAIR_SIZE - 1) * PERTURB_STEP, list_text))
list_text = list(map(lambda x: np.array(x)[:1 + (INFORMATIVE_PAIR_SIZE - 1) * PERTURB_STEP:PERTURB_STEP], list_text))
batch = list_text[-1].tolist()
list_text = np.array(list_text).flatten().tolist()
len(list_text)

df = pd.DataFrame(index=['corr'], columns=dict_meta_data.keys())
for model_name in dict_meta_data.keys():
    model_path = dict_meta_data[model_name]
    
    if 'PromCSE' in model_path or 'promcse' in model_path:
        sys.path.insert(0, 'PromCSE')
        from transformers import AutoConfig
        from promcse.models import RobertaForCL, BertForCL
        import argparse
        config = AutoConfig.from_pretrained(model_path)
        
        # args = argparse.Namespace()
        # args.pre_seq_len = 16
        # args.prefix_projection = False
        # args.prefix_hidden_size = 512
        # args.do_mlm = False
        # args.pooler_type = 'cls'
        # args.temp = 0.05
        # args.do_eh_loss = False
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name_or_path", type=str, 
                default="PromCSE/models/my-unsup-promcse-bert-base-uncased",
                help="Transformers' model name or path")
        parser.add_argument("--pooler", type=str, 
                choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
                default='cls', 
                help="Which pooler to use")
        parser.add_argument("--mode", type=str, 
                choices=['dev', 'test', 'fasttest'],
                default='test', 
                help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
        parser.add_argument("--task_set", type=str, 
                choices=['sts', 'transfer', 'full', 'na'],
                default='na',
                help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
        parser.add_argument("--tasks", type=str, nargs='+', 
                # default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                #          'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                #          'SICKRelatedness', 'STSBenchmark'], 
                default=['STS12'],
                help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
        parser.add_argument("--gpu_id", type=int, default=0,
                help="GPU id to use.")
        ### arguments only for PromCSE 
        parser.add_argument("--pooler_type", type=str, 
                choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
                default='cls', 
                help="Which pooler to use")
        parser.add_argument("--temp", type=float, 
                default=0.05, 
                help="Temperature for softmax.")
        parser.add_argument("--hard_negative_weight", type=float, 
                default=0.0, 
                help="The **logit** of weight for hard negatives (only effective if hard negatives are used).")
        parser.add_argument("--do_mlm", action='store_true', 
                help="Whether to use MLM auxiliary objective.")
        parser.add_argument("--mlm_weight", type=float, 
                default=0.1, 
                help="Weight for MLM auxiliary objective (only effective if --do_mlm).")
        parser.add_argument("--mlp_only_train", action='store_true', 
                help="Use MLP only during training")
        parser.add_argument("--pre_seq_len", type=int, 
                default=16, 
                help="The length of prompt")
        parser.add_argument("--prefix_projection", action='store_true', 
                help="Apply a two-layer MLP head over the prefix embeddings")
        parser.add_argument("--prefix_hidden_size", type=int, 
                default=512, 
                help="The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used")
        parser.add_argument("--do_eh_loss", 
                action='store_true',
                help="Whether to add Energy-based Hinge loss")
        parser.add_argument("--eh_loss_margin", type=float, 
                default=None, 
                help="The margin of Energy-based Hinge loss")
        parser.add_argument("--eh_loss_weight", type=float, 
                default=None, 
                help="The weight of Energy-based Hinge loss")
        parser.add_argument("--cache_dir", type=str, 
                default=None,
                help="Where do you want to store the pretrained models downloaded from huggingface.co")
        parser.add_argument("--model_revision", type=str, 
                default="main",
                help="The specific model version to use (can be a branch name, tag name or commit id).")
        parser.add_argument("--use_auth_token", action='store_true', 
                help="Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models).")
        args = parser.parse_args()
        
        model = BertForCL.from_pretrained(
                model_path,
                from_tf=bool(".ckpt" in model_path),
                config=config,
                # revision=args.model_revision,
                # use_auth_token=True if args.use_auth_token else None,
                model_args=args,
                data_args=None,
        )
    else:
        model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # if torch.cuda.device_count() > 1:
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
        model = DataParallel(model)
        batch_size = BATCH_SIZE * torch.cuda.device_count()
    else:
        batch_size = int(BATCH_SIZE // INFORMATIVE_PAIR_SIZE)
    _ = model.to(DEVICE)

    list_batch = [list_text[i:i + batch_size * INFORMATIVE_PAIR_SIZE] for i in range(0, len(list_text), batch_size * INFORMATIVE_PAIR_SIZE)]

    def func(batch):
        tokenized_input = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        tokenized_input = {k: v.to(DEVICE) for k, v in tokenized_input.items()}
        if 'PromCSE' in model_path or 'promcse' in model_path: # if encoder is PromCSE
            output = model(**tokenized_input, output_hidden_states=True, return_dict=True, sent_emb=True, output_attentions=True)
        else: # if encoder is the one of the others    
            output = model(**tokenized_input, output_hidden_states=True, return_dict=True)
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

df = df.astype(float).round(2)
df
# df.to_csv(f"../sdnorm_dropout_{FILE_NAME.split('.')[0].split('_')[-1]}.csv")