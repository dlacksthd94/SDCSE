import itertools

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
