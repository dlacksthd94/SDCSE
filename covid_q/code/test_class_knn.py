import os
import operator
import numpy as np
from scipy import spatial
from methods import read_pickle, read_csv, write_dict_to_csv


def get_id_to_questions(input_file):
    result = {}     # Dictionary where key is ID, value is list of questions
    input_data = read_csv(input_file, True)   # 2D array where first column is question, second column is question ID

    for row in input_data:
        current_id = int(row[1])
        if current_id not in result:
            result[current_id] = []

        result.get(current_id).append(row[0])

    return result


def get_questions_to_id(input_file):
    result = {}     # Dictionary where key is question, value is ID
    input_data = read_csv(input_file, True)

    for row in input_data:
        result[row[0]] = int(row[1])

    return result


def get_k_nearest_neighbor(test_question, training_questions, question_to_embedding, k, training_questions_to_id, distance_measurement):
    distance_dict = {}      # Dictionary where key is question and value is distance
    test_embedding = question_to_embedding[test_question]

    for training_question in training_questions:
        training_embedding = question_to_embedding[training_question]
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
    ids = []

    for i in range(len(sorted_distance_dict)):
        if len(results) < k:
            current_prediction = sorted_distance_dict[i][0]
            current_id = training_questions_to_id[current_prediction]

            if current_id not in ids:
                results.append(current_prediction)
                ids.append(current_id)
        else:
            break

    return results


def is_correct(training_questions_to_id, predicted_neighbors, question_id):
    for prediction in predicted_neighbors:
        if training_questions_to_id[prediction] is question_id:
            return True

    return False


def get_accuracy(test_id_to_questions, embedding_path, training_question_to_id_path, k):
    question_to_embedding = read_pickle(embedding_path)
    training_questions_to_id = get_questions_to_id(training_question_to_id_path)
    training_questions = training_questions_to_id.keys()

    accuracies_by_id = []
    test_to_predictions = {}

    total_correct, total = 0, 0

    for question_id in test_id_to_questions.keys():
        current_correct = 0
        current_total = 0

        current_questions = test_id_to_questions.get(question_id)
        for test_question in current_questions:
            predicted_neighbors = get_k_nearest_neighbor(test_question, training_questions, question_to_embedding, k, training_questions_to_id, distance_measurement = 'Cosine')
            test_to_predictions[test_question] = predicted_neighbors

            if is_correct(training_questions_to_id, predicted_neighbors, question_id):
                current_correct += 1
            else:
                # print(f"Test: {test_question} | Predictions: {predicted_neighbors}")
                pass

            current_total += 1

        id_accuracy = 1.0*current_correct/current_total
        
        total_correct += current_correct
        total += current_total

        accuracies_by_id.append(id_accuracy)

    write_dict_to_csv('predictions.csv', test_to_predictions)

    return round(np.mean(accuracies_by_id), 3), round(1.0*total_correct/total, 3) # Classwise accuracy, Overall accuracy


#           MAIN            #
for input_test_set in ['dataset_classes/split_3/testA.csv']:
    # print(input_test_set)
    for n in [1, 5]:
        print('k =', n)
        test_id_to_questions = get_id_to_questions(input_test_set)

        # 이 .py 말고 downstream_task.ipynb로 실행할 때의 기준
        directory = 'data/'
        result_classwise = []
        result_overall = []

        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".pickle"):
                classwise_acc, overall_acc = get_accuracy(test_id_to_questions, os.path.join(directory, filename), 'dataset_classes/split_3/train3.csv', n)
                result_classwise.append(classwise_acc)
                result_overall.append(overall_acc)
                print(filename.split('.')[0].split('question_embeddings_')[1], ':', classwise_acc, '/', overall_acc)
        
        # 엑셀에 기록하기 좋게 print
        print()
        print('[For Excel: Classwise Accuracy]')
        for i in [0, 7, 8, 10, 11, 6, 9, 14, 15, 12, 13, 1, 2, 3, 4, 5]:
            print(result_classwise[i])
        print()
        print('[For Excel: Overall Accuracy]')
        for i in [0, 7, 8, 10, 11, 6, 9, 14, 15, 12, 13, 1, 2, 3, 4, 5]:
            print(result_overall[i])
        print()

