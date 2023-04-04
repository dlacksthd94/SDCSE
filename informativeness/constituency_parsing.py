import spacy
from spacy import displacy
from tqdm import tqdm
import joblib as jl
import pickle
import numpy as np
import os
import benepar
import argparse
import torch
import subprocess
import signal
import time

# arg
parser = argparse.ArgumentParser()
parser.add_argument('-p', required=False, help='select constituency parser from [base, large]', type=str, choices=['base', 'large'], default='large')
parser.add_argument('-pp', required=False, help='select constituency parser pipeline from [sm, md, lg]', type=str, choices=['sm', 'md', 'lg'], default='lg')
parser.add_argument('-d', required=False, help='select dataset from [wiki1m, STS12, STS13, STS14, STS15, STS16, STS-B, SICK-R, nli, quora, simplewiki, specter, covid]', type=str, choices=['wiki1m', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R', 'nli', 'quora', 'simplewiki', 'specter', 'covid', 'huffpost'], default='wiki1m')
parser.add_argument('-g', required=False, help='select which gpu to run on', type=int, choices=[0, 1, 2, 3], default=0)
parser.add_argument('-s', required=False, help='automatically split data and run subprocesses in parallel', action='store_true', default=False)
parser.add_argument('-r', required=False, help='indicates the rank of parallel subprocess, assigned by the main process. -1 indicates no parallelism', type=int, default=-1)
args = parser.parse_args()

n_gpu = torch.cuda.device_count()
gpu_ram_capacity = torch.cuda.get_device_properties(0).total_memory / 1024**3 # in GB
num_proc_available_in_one_device = int(gpu_ram_capacity // 5)

DATASET = args.d
PARSER = 'benepar_en3' if args.p == 'base' else 'benepar_en3_large'
PIPELINE = f'en_core_web_{args.pp}'
N = '_' + str(args.r) if args.r != -1 else ''
print(N)
print(DATASET)

# load dataset
with open(f'../SimCSE/{DATASET}.txt') as f:
    list_text = f.readlines()

# preprocessing
for i in tqdm(range(len(list_text))):
    list_text[i] = list_text[i].strip('\n.')

# load model
if not args.s:
    torch.cuda.set_device(args.g)
    
    benepar.download(PARSER)
    model = spacy.load(PIPELINE)

    if spacy.__version__.startswith('2'):
        model.add_pipe(benepar.BeneparComponent(PARSER))
    else:
        model.add_pipe("benepar", config={"model": PARSER})

# start parsing
len_text = len(list_text)
num_total_proc = n_gpu * num_proc_available_in_one_device if args.r != -1 or args.s else 1
window = len_text // num_total_proc
list_window = list(range(0, len_text, window)) + [len_text]
list_range = list(zip(list_window[:-1], list_window[1:]))

# parse sentence
if args.s:
    list_subproc = [subprocess.Popen(f'python constituency_parsing.py -p large -pp lg -d wiki1m -r {rank} -g {rank // n_gpu}', shell=True, preexec_fn=os.setsid) for rank, range_temp in enumerate(list_range)]
    list_wait = [subproc.wait() for subproc in list_subproc]
    # [os.killpg(os.getpgid(subproc.pid), signal.SIGTERM) for subproc in list_subproc] # in case if you want to terminate all subprocesses
else:
    if not os.path.exists(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle'):
        list_tree = []
        start_index, end_index = list_range[int(args.r)]
        # for text in tqdm(list_text[start_index:end_index], initial=start_index, total=len_text):
        for text in tqdm(list_text[start_index:end_index], initial=start_index, total=len_text):
            try:
                doc = model(text)
                list_tree.append(doc)
            except:
                list_tree.append([])

        with open(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'wb') as f:
            pickle.dump(list_tree, f)
    else:
        print(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle already exists!')
        with open(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}{N}.pickle', 'rb') as f:
            list_tree = pickle.load(f)

# concat results
if args.s:
    # while sum([list_subproc[0].poll() is None for subproc in list_subproc]):
    #     time.sleep(10)
    list_tree = []
    for i in tqdm(range(0, num_total_proc)):
        try:
            with open(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}_{i}.pickle', 'rb') as f:
                list_tree_temp = pickle.load(f)
            list_tree.extend(list_tree_temp)
        except:
            pass
    len(list_tree)
    
    print('saving', f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}.pickle', '\n', 'can take some time')
    with open(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}.pickle', 'wb') as f:
        pickle.dump(list_tree, f)

# get depth
def walk_tree_cst(node, depth):
    if list(node._.children):
        return max([walk_tree_cst(child, depth + 1) for child in node._.children])
    else:
        return depth

if args.r == -1 or args.s:
    list_depth = []
    for doc in tqdm(list_tree):
        if len(doc):
            depth = walk_tree_cst(list(doc.sents)[0], 0)
            # [walk_tree_cst(sent.root, 0) for sent in doc.sents]
        else:
            depth = 0
        list_depth.append(depth)

    with open(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}{N}_depth.pickle', 'wb') as f:
        pickle.dump(list_depth, f)