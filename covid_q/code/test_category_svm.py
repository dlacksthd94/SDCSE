# Tests BERT embeddings with an SVM model

import os
from methods import read_pickle, read_csv, write_dict_to_csv
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score


def get_embedding_data(input_file, category_to_num, question_to_embedding):
    input_data = read_csv(input_file, False)

    embeddings = []
    labels = []

    for row in input_data:
        if row[1] == '':
            continue
        else:
            embedding = question_to_embedding[row[0]]
            category_number = category_to_num[row[1]]
            embeddings.append(embedding)
            labels.append(category_number)

    return np.asarray(embeddings), np.asarray(labels)


def check_predictions(classifier, embeddings, ground_truth):
    predictions = classifier.predict(embeddings)

    return accuracy_score(ground_truth, predictions)



#       MAIN      #
category_to_num = {'Transmission': 0,
                   'Prevention': 1,
                   'Societal Effects': 2,
                   'Societal Response': 3,
                   'Reporting': 4,
                   'Origin': 5,
                   'Treatment': 6,
                   'Testing': 7,
                   'Comparison': 8,
                   'Individual Response': 9,
                   'Economic Effects': 10,
                   'Speculation': 11,
                   'Having COVID': 12,
                   'Nomenclature': 13,
                   'Symptoms': 14}


# 이 .py 말고 downstream_task.ipynb로 실행할 때의 기준
directory = 'data/'
result = []

for filename in sorted(os.listdir(directory)):
    if filename.endswith(".pickle"):
        question_to_embedding = read_pickle(os.path.join(directory, filename))
        
        train_x, train_y = get_embedding_data('dataset_categories/train20.csv', category_to_num, question_to_embedding)
        testA_x, testA_y = get_embedding_data('dataset_categories/testA.csv', category_to_num, question_to_embedding)
        
        classifier = svm.SVC()
        classifier.fit(train_x, train_y)        
        
        acc = round(check_predictions(classifier, testA_x, testA_y), 3)
        result.append(acc)
        print(filename.split('.')[0].split('question_embeddings_')[1], ':', acc)

# 엑셀에 기록하기 좋게 print
print()
print('[For Excel]')
for i in [0, 7, 8, 10, 11, 6, 9, 14, 15, 12, 13, 1, 2, 3, 4, 5]:
    print(result[i])
