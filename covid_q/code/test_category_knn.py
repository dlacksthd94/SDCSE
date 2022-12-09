import os
import operator
import numpy as np
from scipy import spatial
from methods import read_csv, read_pickle


def get_k_nearest_neighbor(test_embedding, training_questions, question_to_embedding, training_questions_to_category, k, distance_measurement):
    distance_dict = {}      # Dictionary where key is question and value is distance

    for training_question in training_questions:
        training_embedding = question_to_embedding.get(training_question)
        current_distance = 0

        if distance_measurement == 'Euclidean':
            current_distance = np.linalg.norm(list(np.subtract(test_embedding, training_embedding)))
        elif distance_measurement == 'Cosine':
            current_distance = spatial.distance.cosine(list(test_embedding), list(training_embedding))

        if current_distance is not 0:
            distance_dict[training_question] = current_distance
        else:
            print(f"Error, distance is 0 for question: {training_question}")
    
    sorted_distance_dict = sorted(distance_dict.items(), key=operator.itemgetter(1))
    results = []
    categories = []

    for i in range(len(sorted_distance_dict)):
        if len(results) < k:
            current_prediction = sorted_distance_dict[i][0]
            current_category = training_questions_to_category[current_prediction]

            if current_category not in categories:
                results.append(current_prediction)
                categories.append(current_category)
        else:
            break

    return results


def get_questions_to_category(input_path):
    result = {}     # Dictionary where key is question, value is category
    input_data = read_csv(input_path, skip_header=False)

    for row in input_data:
        result[row[0]] = row[1]

    return result


def is_correct(nearest_neighbors, training_questions_to_category, ground_truth):
    predicted_categories = []

    for prediction in nearest_neighbors:
        predicted_categories.append(training_questions_to_category[prediction])

    return ground_truth in predicted_categories


def get_accuracy(testing_questions_to_category, training_questions_to_category, question_to_embeddings):
    training_questions = training_questions_to_category.keys()

    num_correct, num_total = 0, 0

    for test_question in testing_questions_to_category.keys():
        test_embedding = question_to_embeddings[test_question]
        ground_truth = testing_questions_to_category[test_question]

        # k값, Distance measurement 수정 가능
        nearest_neighbors = get_k_nearest_neighbor(test_embedding, training_questions, question_to_embeddings, training_questions_to_category, k = 5, distance_measurement = 'Cosine')

        if is_correct(nearest_neighbors, training_questions_to_category, ground_truth) is True:
            num_correct += 1
        else:
            # print(f"Test: {test_question} | Predictions: {nearest_neighbors}")
            pass
        
        num_total += 1

    return 1.0*num_correct/num_total



#           MAIN            #
training_questions_to_category = get_questions_to_category('dataset_categories/train20.csv')
testing_questions_to_category = get_questions_to_category('dataset_categories/testA.csv')

# 이 .py 말고 downstream_task.ipynb로 실행할 때의 기준
directory = 'data/'
result = []

for filename in sorted(os.listdir(directory)):
    if filename.endswith(".pickle"):
        question_to_embeddings = read_pickle(os.path.join(directory, filename))
        acc = round(get_accuracy(testing_questions_to_category, training_questions_to_category, question_to_embeddings), 3)
        result.append(acc)
        print(filename.split('.')[0].split('question_embeddings_')[1], ':', acc)

# 엑셀에 기록하기 좋게 print
print()
print('[For Excel]')
for i in [0, 7, 8, 10, 11, 6, 9, 14, 15, 12, 13, 1, 2, 3, 4, 5]:
    print(result[i])