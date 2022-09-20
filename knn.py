import math
from turtle import color

import pandas as pd
import scipy.stats
import numpy as np
import sklearn.metrics as sk
from matplotlib import pyplot as plot
import statistics


tweets = pd.read_csv('./transformed_data_1.csv')
# tweets = pd.read_csv('./example.csv')
# tweets.head()
tweets.drop(columns=tweets.columns[0], axis=1, inplace=True)
print(tweets)
# For shuffling the dataset
# abalone = abalone.sample(frac=1)

# For splitting data into 3 sets training, validation, and testing
validation_data, testing_data, training_data = np.split(tweets, [int(.2 * len(tweets)), int(.5 * len(tweets))])
# training_data, validation_data, testing_data = np.split(abalone, [int(.5 * len(abalone)), int(.7 * len(abalone))])

# Extract all features except "Rings" or Output column into one array and
# Rings column in another array for all data sets

last_col_name = str(tweets.shape[1] - 1)

actual_training_data = training_data.drop(last_col_name, axis=1).to_numpy()
training_data_rings = training_data[last_col_name].to_numpy()

actual_validation_data = validation_data.drop(last_col_name, axis=1).to_numpy()
validation_data_rings = validation_data[last_col_name].to_numpy()
# for i in range(( validation_data["Rings"]).length):
#     print(scipy.stats.mode(validation_data["Rings"]))

actual_testing_data = testing_data.drop(last_col_name, axis=1).to_numpy()
testing_data_rings = testing_data[last_col_name].to_numpy()

# initializing empty dictionary for storing each data set accuracies
k_training_accuracy_dict = {}
k_validation_accuracy_dict = {}
k_testing_accuracy_dict = {}



# this generic method will calculate the distance of one sample with actual data (training data) &
# get the K nearest distances and then predict the output class (Rings) and then store the predicted value into array.
# Parameters:-
# sample: one row from the data set
# actual_data: complete dataset from which we want to calculate the distance of our sample
# data_rings: actual output class of dataset(Rings in our case) from which we want to get classes of neighbors
# pred_ring: the array that will contain the predicted value of each sample
# k: the value of k; how much neighbors we want
def calculate_dis_pred(sample, actual_data, data_rings, pred_ring, k):
    distances = (np.linalg.norm(actual_data - sample, axis=1))
    nearest_neighbor_ids = distances.argsort()[:k]
    nearest_neighbor_rings = data_rings[nearest_neighbor_ids]

    prediction = statistics.mode(nearest_neighbor_rings)

    pred_ring.append(prediction)


#  ********** Part - 1 **************
# Loop to find the tuned value of k, in this loop, for each value of k, it will find the predicted value of each
# sample of each dataset and after that find the accuracy of predicted classes with actual classes and store it
# in the dictionary with the key k
def n_times_knn(n):
    for k in range(1, n, 2):
        pred_training_rings = []
        pred_validation_rings = []
        pred_testing_rings = []
        for training_sample in actual_training_data:
            calculate_dis_pred(training_sample, actual_training_data, training_data_rings, pred_training_rings, k)
        k_training_accuracy_dict[k] = sk.accuracy_score(training_data_rings, pred_training_rings)

        for v_sample in actual_validation_data:
            calculate_dis_pred(v_sample, actual_training_data, training_data_rings, pred_validation_rings, k)
        k_validation_accuracy_dict[k] = sk.accuracy_score(validation_data_rings, pred_validation_rings)

        for test_sample in actual_testing_data:
            calculate_dis_pred(test_sample, actual_training_data, training_data_rings, pred_testing_rings, k)
        k_testing_accuracy_dict[k] = sk.accuracy_score(testing_data_rings, pred_testing_rings)

    print(k_training_accuracy_dict)
    return k_training_accuracy_dict, k_validation_accuracy_dict, k_testing_accuracy_dict


# This generic method will calculate the Accuracy, Precision, Recall, and F1 Score of specific k value.
# Parameters:-
# k: no. of neighbors
# data_set: dataset from which we want to make calculations
# data_set_rings: actual output class of that given data_set
# _type: to specify the type of data_set
def get_calculation(k, data_set, data_set_rings, _type):
    data_pred = []
    for sample in data_set:
        calculate_dis_pred(sample, data_set, data_set_rings, data_pred, k)

    accuracy = sk.accuracy_score(data_set_rings, data_pred)
    recall = sk.recall_score(data_set_rings, data_pred, average='weighted')
    precision = sk.precision_score(data_set_rings, data_pred, average='weighted')
    f1 = sk.f1_score(data_set_rings, data_pred, average='weighted')

    print('-' * 10 + f'Calculation On K = {k}' + '-' * 10)
    print(f'On {_type} Data:')
    print(f'Accuracy on K = {k} is "{accuracy}"')
    print(f'Precision on K = {k} is "{precision}"')
    print(f'Recall on K = {k} is "{recall}"')
    print(f'F1 Score on K = {k} is "{f1}"')


#  ********** Part - 2 and Part - 3 **************
# As we got best result at k = 18.2642 and that is even number so we take 17 and 19
# odd number to check the calculations

# k = math.ceil(math.sqrt(len(training_data)))
# k = k + 1 if k % 2 == 0 else k
k = 61
get_calculation(k, actual_training_data, training_data_rings, 'training')
get_calculation(k, actual_testing_data, testing_data_rings, 'testing')

# k = k - 1 if k % 2 == 0 else k
# get_calculation(k, actual_training_data, training_data_rings, 'training')
# get_calculation(k, actual_testing_data, testing_data_rings, 'testing')


#  ********** Part - 4 **************
def plot_accuracy_graph():
    plot.title('Training Accuracy vs Testing Accuracy')
    plot.xlabel('n_neighbor')
    plot.ylabel('Accuracy')

    # As we have accuracy on y-axis, so it takes values from the dictionary
    # As we have k (n_neighbors) on x-axis, so it takes keys from dictionary because keys are the value of k
    # Called a built-in method that take x and y axis
    # first argument is x-axis (k-values) and second is y-axis (accuracies)
    # this is for training dataset
    plot.plot(list(k_training_accuracy_dict.keys()), list(k_training_accuracy_dict.values()), color='green')

    # this is for validation dataset
    plot.plot(list(k_validation_accuracy_dict.keys()), list(k_validation_accuracy_dict.values()), color='brown')

    # this is for testing dataset
    plot.plot(list(k_testing_accuracy_dict.keys()), list(k_testing_accuracy_dict.values()), color='red')
    plot.show()


# k_training_accuracy_dict, k_validation_accuracy_dict, k_testing_accuracy_dict = n_times_knn(20)
# plot_accuracy_graph()
