import torch
import numpy as np
import csv
from transformers import BertModel, BertTokenizer
from methods import clean_line, save_to_pickle, read_csv, read_pickle

# SBERT에서 사용
def normalize(embeddings):
    return embeddings / np.linalg.norm(embeddings)

# Encode text
def get_embedding(input_string, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(input_string, add_special_tokens = True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0].numpy()  # Models outputs are now tuples
        # last_hidden_states = last_hidden_states[:, 0, :]
        last_hidden_states = np.mean(last_hidden_states, axis = 1)
        # last_hidden_states = np.amax(last_hidden_states, axis = 1)
        last_hidden_states = last_hidden_states.flatten()
        return last_hidden_states.tolist()

# [BERT] Gets all encodings given an input csv
def get_all_embeddings(input_csv, tokenizer, model):
    result = {}     # Dictionary where key = question and value = bert embedding for that question

    reader = read_csv(input_csv, True)

    for row in reader:
        question = row[2]
        embedding = get_embedding(question, tokenizer, model)
        question = ''.join([i if ord(i) < 128 else ' ' for i in question])

        result[question] = embedding

    return result

# [SBERT] Gets all encodings given an input csv
def get_all_embeddings_sbert(input_csv, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    result = {}     # Dictionary where key = question and value = bert embedding for that question
    reader = read_csv(input_csv, True)

    for row in reader:
        question = row[2]
        embedding = normalize(model.encode(question)).tolist()
        question = ''.join([i if ord(i) < 128 else ' ' for i in question])

        result[question] = embedding

    return result

# [SimCSE] Gets all encodings given an input csv
def get_all_embeddings_simcse(input_csv, model_name):
    from simcse import SimCSE
    model = SimCSE(model_name)

    result = {}     # Dictionary where key = question and value = bert embedding for that question
    reader = read_csv(input_csv, True)

    for row in reader:
        question = row[2]
        embedding = model.encode(question).tolist()
        question = ''.join([i if ord(i) < 128 else ' ' for i in question])

        result[question] = embedding

    return result

# [DiffCSE] Gets all encodings given an input csv
def get_all_embeddings_diffcse(input_csv, model_name):
    import sys
    sys.path.append('../../')

    from DiffCSE.diffcse import DiffCSE
    model = DiffCSE(model_name)

    result = {}     # Dictionary where key = question and value = bert embedding for that question
    reader = read_csv(input_csv, True)

    for row in reader:
        question = row[2]
        embedding = model.encode(question).tolist()
        question = ''.join([i if ord(i) < 128 else ' ' for i in question])

        result[question] = embedding

    return result


def combine_with_augmented_dataset(original_pickle_path, new_pickle_path, augmented_dataset_path, tokenizer, model):
    original_data = read_pickle(original_pickle_path)
    augmented_data = read_csv(augmented_dataset_path, skip_header = False)

    for row in augmented_data:
        question = row[0]
        embedding = get_embedding(question, tokenizer, model)
        question = ''.join([i if ord(i) < 128 else ' ' for i in question])

        original_data[question] = embedding

    save_to_pickle(original_data, new_pickle_path)

    return original_data


#           MAIN            #
input_csv = '../data/final_master_dataset_1599.csv'
which_model_to_use = 'DiffCSE' # 모델 선택

if which_model_to_use == 'BERT':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    all_embeddings = get_all_embeddings(input_csv, tokenizer, model)
    save_to_pickle(all_embeddings, '../data/question_embeddings_pooled.pickle')

elif which_model_to_use == 'SBERT':

    model_list = [   
        'nli-bert-base',
        'nli-roberta-base',
        'stsb-bert-base',
        'stsb-roberta-base',
        'bert-base-nli-stsb-mean-tokens',
        'roberta-base-nli-stsb-mean-tokens'
        ]
    
    for model_name in model_list:
        all_embeddings = get_all_embeddings_sbert(input_csv, model_name)
        save_to_pickle(all_embeddings, f'../data/question_embeddings_sbert_{model_name}.pickle')
        print(f'Successfully saved question_embeddings_sbert_{model_name}.pickle')

elif which_model_to_use == 'SimCSE':

    model_list = [   
        'princeton-nlp/unsup-simcse-bert-base-uncased',
        'princeton-nlp/unsup-simcse-roberta-base',
        'princeton-nlp/sup-simcse-bert-base-uncased',
        'princeton-nlp/sup-simcse-roberta-base'
        ]
    
    for model_name in model_list:
        all_embeddings = get_all_embeddings_simcse(input_csv, model_name)
        model_name_split = model_name.split('/')[1]
        save_to_pickle(all_embeddings, f'../data/question_embeddings_simcse_{model_name_split}.pickle')
        print(f'Successfully saved question_embeddings_simcse_{model_name_split}.pickle')

elif which_model_to_use == 'DiffCSE':

    model_list = [   
        'voidism/diffcse-bert-base-uncased-sts',
        'voidism/diffcse-bert-base-uncased-trans',
        'voidism/diffcse-roberta-base-sts',
        'voidism/diffcse-roberta-base-trans'
        ]
    
    for model_name in model_list:
        all_embeddings = get_all_embeddings_diffcse(input_csv, model_name)
        model_name_split = model_name.split('/')[1]
        save_to_pickle(all_embeddings, f'../data/question_embeddings_diffcse_{model_name_split}.pickle')
        print(f'Successfully saved question_embeddings_diffcse_{model_name_split}.pickle')

# PromCSE는 PromCSE 폴더의 get_promcse_embeddings.py 사용