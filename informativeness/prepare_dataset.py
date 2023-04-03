import itertools
import jsonlines
import csv
import pandas as pd
import numpy as np

N = 1000000

np.random.seed(1000)

dict_sts_dataset = {
    'STS12': ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews'], 
    'STS13': ['FNWN', 'headlines', 'OnWN'], 
    'STS14': ['deft-forum', 'deft-news', 'headlines', 'images', 'OnWN', 'tweet-news'], 
    'STS15': ['answers-forums', 'answers-students', 'belief', 'headlines', 'images'], 
    'STS16': ['answer-answer', 'headlines', 'plagiarism', 'postediting', 'question-question']
}
for dataset in dict_sts_dataset:
    list_text = []
    for source in dict_sts_dataset[dataset]:
        with open(f'../SimCSE/SentEval/data/downstream/STS/{dataset}-en-test/STS.input.{source}.txt', 'r') as f:
            list_line = f.readlines()
        list_text_temp = list(itertools.chain.from_iterable(map(lambda line: line.strip().split('\t'), list_line)))
        list_text.extend(list_text_temp)
    with open(f'../SimCSE/{dataset}.txt', 'w') as f:
        _ = f.write('\n'.join(list_text))
    
with open(f'../SimCSE/SentEval/data/downstream/STS/STSBenchmark/sts-test.csv', 'r') as f:
    list_line = f.readlines()
list_text = list(itertools.chain.from_iterable(map(lambda line: line.strip().split('\t')[5:7], list_line)))
with open(f'../SimCSE/STS-B.txt', 'w') as f:
    _ = f.write('\n'.join(list_text))
    
with open(f'../SimCSE/SentEval/data/downstream/SICK/SICK_test_annotated.txt', 'r') as f:
    list_line = f.readlines()
list_text = list(itertools.chain.from_iterable(map(lambda line: line.strip().split('\t')[1:3], list_line)))
with open(f'../SimCSE/SICK-R.txt', 'w') as f:
    _ = f.write('\n'.join(list_text))

with open(f'../SimCSE/wiki1m_for_simcse.txt', 'r') as f:
    list_line = f.readlines()
list_text = list(map(str.strip, list_line))
if N != 1000000:
    np.random.shuffle(list_text)
list_text = list_text[:N]
with open(f'../SimCSE/wiki1m.txt', 'w') as f:
    _ = f.write('\n'.join(list_text))

dict_dataset = {'SimpleWiki': 'simplewiki', 'specter_train_triples': 'specter'}
for dataset in dict_dataset:
    with jsonlines.open(f'../SimCSE/{dataset}.jsonl', 'r') as f:
        list_text = list(itertools.chain.from_iterable(f))
    np.random.shuffle(list_text)
    list_text_sampled = list_text[:N]
    with open(f'../SimCSE/{dict_dataset[dataset]}.txt', 'w') as f:
        _ = f.write('\n'.join(list_text_sampled))

with jsonlines.open(f'../SimCSE/quora_duplicates_triplets.jsonl', 'r') as f:
    list_text_temp = []
    for line in f:
        query, pos, neg = dict(line).values()
        list_text_temp.append([query] + pos + neg)
list_text = list(itertools.chain.from_iterable(list_text_temp))
np.random.shuffle(list_text)
list_text_sampled = list_text[:N]
with open(f'../SimCSE/quora.txt', 'w') as f:
    _ = f.write('\n'.join(list_text_sampled))

with open(f'../SimCSE/final_master_dataset.csv', 'r') as f:
    list_line = list(csv.reader(f))
list_text = list(map(lambda line: line[2] + '?', list_line))
with open(f'../SimCSE/covid.txt', 'w') as f:
    _ = f.write('\n'.join(list_text))
    
df_huffpost = pd.read_csv(f'../SimCSE/huffpost_positive_pairs_{N}.csv')
df_huffpost['headline'].to_csv(f'../SimCSE/huffpost.txt', index=False, header=False)