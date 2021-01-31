import math
import operator
def data_process():
    return 0

def get_distance(datapoint1, datapoint2):
    distance = 0
    for x in range(4):
        distance += (datapoint1[x] - datapoint2[x])**2

    final_distance = math.sqrt(distance)
    return final_distance

def get_K_neighbors(train_dataset, test_datapoint, K):
    ## 1. get all distances
    all_distances = []
    for i in range(len(train_dataset)):
        distance = get_distance(train_dataset[i], test_datapoint)
        all_distances.append((train_dataset[i], distance))

    ### 2. sort all distances
    print(all_distances)
    print(operator.itemgetter(1))
    all_distances.sort(key=operator.itemgetter(1))
    print(all_distances)

    ### 3. pickup the K nearest neighbours
    K_neighbors = []
    for k in range(K):
        K_neighbors.append(all_distances[k][0])
        print(K_neighbors)

    return K_neighbors

def get_prediction(K_neighbors):
    category_votes = {}
    for i in range(len(K_neighbors)):
        category = K_neighbors[i][-1]
        if category in category_votes:
            category_votes[category] +=1
        else:
            category_votes[category] = 1
    sort_category_votes = sorted(category_votes.items(), key=operator.itemgetter(1), reverse=True)

    most_vote = sort_category_votes[0][0]
    return most_vote

def simple_case_study():

    ##### stage 1 :   data preparation
    train_dataset = [[2, 2, 2, 2, 'a'],
                     [4, 4, 4, 4, 'b'],
                     [1, 1, 1, 1, 'a'],
                     [3, 3, 3, 3, 'b']]

    test_datapoint = [5, 5, 5, 5]   # ? a or b

    ######  stage 2:  alrogithm
    K = 4
    K_neighbors = get_K_neighbors(train_dataset, test_datapoint, K)
    print('K neighbors:', K_neighbors)

    prediction = get_prediction(K_neighbors)
    print('final prediction of the test datapoint is:', prediction)



################################
if __name__ == "__main__":
    simple_case_study()