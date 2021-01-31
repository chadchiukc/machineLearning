import pandas as pd
import random
import math
import operator

K = 20
PARAM_NUM = 4
PERCENTAGE_OF_TRAINING = 0.8


def get_accuracy():
    epoch = 0
    overall_accuracy = 0
    while epoch < 10:
        ##Load the txt data file and append to a list
        iris_data = pd.read_csv("iris.data.txt", header=None)
        full_dataset = [list(row) for row in iris_data.values]

        ##shuffle data set and separate dataset to training and testing
        random.shuffle(full_dataset)
        train_dataset = full_dataset[: int(len(full_dataset) * PERCENTAGE_OF_TRAINING)]
        test_dataset = full_dataset[int(len(full_dataset) * PERCENTAGE_OF_TRAINING):]

        ###Calculate accuracy
        success_prediction_num = 0
        for test_datapoint in test_dataset:
            K_neighbors = get_K_neighbors(train_dataset, test_datapoint, K, PARAM_NUM)
            # print(test_datapoint[-1], get_prediction(K_neighbors))
            if test_datapoint[-1] == get_prediction(K_neighbors):
                success_prediction_num += 1
        accuracy = success_prediction_num / len(test_dataset)
        print("Accuracy: %0.3f" % accuracy)
        overall_accuracy += accuracy
        epoch += 1
    overall_accuracy = overall_accuracy / epoch
    print("overall accuracy: %0.3f" % overall_accuracy)


###Input your own data and get the prediction
def input_data():
    ##Load the txt data file and append to a list
    iris_data = pd.read_csv("iris.data.txt", header=None)
    full_dataset = [list(row) for row in iris_data.values]

    ###Data Input
    param_1 = float(input("Enter first value: "))
    param_2 = float(input("Enter second value: "))
    param_3 = float(input("Enter third value: "))
    param_4 = float(input("Enter fourth value: "))
    input_datapoint = [param_1, param_2, param_3, param_4]

    ###Get Prediction
    K_neighbors = get_K_neighbors(full_dataset, input_datapoint, K, PARAM_NUM)
    print('Prediction: {}'.format(get_prediction(K_neighbors)))


def get_distance(datapoint1, datapoint2, PARAM_NUM):
    distance = 0
    for x in range(PARAM_NUM):
        distance += (datapoint1[x] - datapoint2[x]) ** 2

    final_distance = math.sqrt(distance)
    return final_distance


def get_K_neighbors(train_dataset, test_datapoint, K, PARAM_NUM):
    ## 1. get all distances
    all_distances = []
    for i in range(len(train_dataset)):
        distance = get_distance(train_dataset[i], test_datapoint, PARAM_NUM)
        all_distances.append((train_dataset[i], distance))
    # print(all_distances)

    ### 2. sort all distances
    all_distances.sort(key=operator.itemgetter(1))

    ### 3. pickup the K nearest neighbours
    K_neighbors = []
    for k in range(K):
        K_neighbors.append(all_distances[k][0])
        # print(type(all_distances[k]))

    return K_neighbors


def get_prediction(K_neighbors):
    category_votes = {}
    for i in range(len(K_neighbors)):
        category = K_neighbors[i][-1]
        if category in category_votes:
            category_votes[category] += 1
        else:
            category_votes[category] = 1
    sort_category_votes = sorted(
        category_votes.items(), key=operator.itemgetter(1), reverse=True
    )

    most_vote = sort_category_votes[0][0]
    return most_vote


################################
if __name__ == "__main__":
    get_accuracy()
    # input_data()
