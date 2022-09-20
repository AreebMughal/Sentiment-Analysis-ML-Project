from distutils.log import Log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from matplotlib import pyplot as plot


def get_initial_thetas(features_length):
    thetas_array = np.zeros((features_length,))

    for i in range(features_length):
        random_num = random.uniform(0, 1)
        random_num = round(random_num, 5)
        thetas_array[i] = random_num

    return pd.Series(thetas_array)


def sigmoid(x):
    # ex = math.exp
    return 1 / (1 + math.exp(-x))


def hypothesis(X, theta):
    return np.dot(pd.Series(X), theta)


def derivative(hypMinusY, X_tr):
    return np.dot(X_tr.T, hypMinusY) / len(X_tr)


def cost(x, hyp, y):
    y_log = np.dot(y.T, np.log(np.array(hyp)))
    one_minus_y = (1 - pd.Series(y)).to_numpy()
    one_minus_hyp = (1 - pd.Series(hyp)).to_numpy()
    # print(one_minus_y)
    # print(one_minus_hyp)
    one_minus_y_log = np.dot(one_minus_y.T, np.log(one_minus_hyp))
    # return (-1 / len(x)) * np.sum((np.dot(y.T, np.log(hyp)), np.dot((1 - y).T, np.log(1 - hyp))))
    return (-1.0 / len(x)) * (y_log + one_minus_y_log)


def regression(classes, thetas, _Y, _X):

    all_classes_sig_res = {}
    h = []
    for _class in classes:
        # sigResult = pd.Series(np.zeros((len(y_tr), )), dtype='float64')
        sigResult = []
        # hypMinusY = pd.Series([], dtype='float64')
        # print('Class', int(_class))
        df = pd.DataFrame(_Y, dtype='int')
        last_col_name = df.shape[1] - 1
        df[last_col_name] = np.where(df[last_col_name] == _class, 1, 0)
        y = df[last_col_name].to_numpy()
        for index, X in enumerate(_X):
            z = hypothesis(X, thetas)
            # print(index, 'z', sigmoid(z))
            # sigResult[index] = sigmoid(z)
            sigResult.append(sigmoid(z))
        # print(sigResult)
        h = sigResult
        all_classes_sig_res[_class] = np.subtract(sigResult, y)
    # print(all_classes_sig_res)
    y_prime = []
    for i in range(len(_Y)):
        _min_arr = []
        for label in classes:
            # print(label)
            _min_arr.append(all_classes_sig_res[label][i])
        min_ele = min(_min_arr)
        index = _min_arr.index(min_ele)
        pred = classes[index]
        y_prime.append(pred)
    return h, y_prime



def Logistic_Regression(n, _thetas):
    global classes
    # errors_in_n = {}
    training_errors = []
    validation_errors = []
    J = 0.0
    alpha = 0.2

    for i in range(n):
        old_thetas = _thetas
        h, y_prime = regression(classes, _thetas, y_tr, X_tr)
        # hypMinusY = np.subtract(y_prime, y_r)
        hypMinusY = np.subtract(h, y_tr)
        J = cost(X_tr, h, y_tr)
        print('Train Error ', J)

        # print(type(new_thetas), type(thetas))
        # print(new_thetas)
        # print(thetas)
        # training_errors[str(i + 1)] = J
        training_errors.append(J)

        h, y_prime = regression(classes, _thetas, y_vld, X_vld)
        # hypMinusY = np.subtract(y_prime, y_r)
        hypMinusY = np.subtract(h, y_vld)
        J = cost(X_vld, h, y_vld)
        print('Valid Error ', J)
        validation_errors.append(J)

        d = derivative(hypMinusY, X_tr)
        _thetas = old_thetas - (alpha * d)
        if old_thetas.equals(_thetas):
            return J, _thetas

    return J, _thetas, training_errors, validation_errors


if __name__ == '__main__':
    no_of_samples = 100
    tweets = pd.read_csv('./transformed_data_1.csv')
    # tweets = pd.read_csv('./example.csv')
    tweets.drop(columns=tweets.columns[0], axis=1, inplace=True)

    x_zero = np.ones((tweets.shape[0]), dtype='int16')
    tweets.insert(0, '', x_zero)
    # print(tweets)
    tweets = tweets[:no_of_samples]
    percent = math.floor(.7 * no_of_samples)
    # print(percent)
    training_data = tweets[:percent]
    testing_data = tweets[percent:]
    training_data.dropna(inplace=True)

    valid_percent = math.floor(.5 * training_data.shape[0])
    train = training_data[:valid_percent]
    validation = training_data[valid_percent:]

    last_col_name = str(tweets.shape[1] - 2)
    X_tr = train.drop(last_col_name, axis=1)
    thetas_length = int(X_tr.shape[1])

    X_tr = X_tr.to_numpy()
    y_tr = train[last_col_name].to_numpy()
    classes = np.unique(y_tr)
    X_vld = validation.drop(last_col_name, axis=1).to_numpy()
    y_vld = validation[last_col_name].to_numpy()

    thetas = get_initial_thetas(thetas_length)
    print('OLD Thetas')
    print(thetas)
    n = 20
    J, thetas, train_error_dict, valid_error_dict = Logistic_Regression(n, thetas)
    print('New Thetas')
    print(thetas)

    print(train_error_dict)

    X_te = testing_data.drop(last_col_name, axis=1).to_numpy()
    y_te = testing_data[last_col_name].to_numpy()

    h, y_prime = regression(classes, thetas, y_te, X_te)
    # hypMinusY = np.subtract(y_prime, y_r)
    hypMinusY = np.subtract(h, y_te)
    J = cost(X_te, h, y_te)
    print()
    print('Test Error:', J)

    def plot_accuracy_graph():
        plot.title('No. of Iteration vs Error')
        plot.xlabel('No. of Iterations')
        plot.ylabel('Error')
        plot.plot([i for i in range(1, n+1)], train_error_dict, color='green')
        plot.plot([i for i in range(1, n + 1)], valid_error_dict, color='brown')
        plot.show()


    plot_accuracy_graph()
