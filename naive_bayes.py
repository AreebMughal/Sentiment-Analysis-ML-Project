import csv

import pandas as pd
import numpy as np
import math
import sklearn.metrics as sk
from main import data_preprocessing, my_transformation, find_unique_words


# ---------------------- Method-1 ----------------------
def get_word_count_in_class(data, word, indices):
    count = 0
    for _index in indices:
        line = data[_index]
        count += sum([1 for wrd in line.split() if wrd == word])
    return count


def probability_a_given_b(sample, _class, data):
    global indices_N_counter_dict
    global all_unique_words

    mle_probability = 1.0
    for word in sample.split():
        count = get_word_count_in_class(data, word, indices_N_counter_dict[_class]['indices'])
        prob = probability(count, indices_N_counter_dict[_class]['words_count'], all_unique_words)
        mle_probability *= prob
        # print(word, '=', count, 'Probability', mle_probability)
    return mle_probability


def posterior_probability(_class, data):
    global indices_N_counter_dict
    # wrong ...
    return indices_N_counter_dict[_class]['words_count'] / len(data)


def append_index_N_count(data_dict, transformed_data, i):
    data_dict['indices'].append(i)
    data_dict['words_count'] += sum(transformed_data[i])


def compute_index_N_count(output_column):
    global classes
    global indices_N_counter_dict
    for index, value in enumerate(output_column):
        if value in classes:
            append_index_N_count(indices_N_counter_dict[value], transformed_samples, index)

    return indices_N_counter_dict
    # elif value == -1:
    #     append_index_N_count(indices_N_counter_dict[value], transformed_samples, index)
    # elif value == 0:
    #     append_index_N_count(indices_N_counter_dict[value], transformed_samples, index)


# --------------------------------------------------------------------------

#  let n be the number of words in the _class case
#  word_k the number of times word k occur
def probability(word_k, n, vocab, laplace_lambda=1):
    return (word_k + laplace_lambda) / (n + (len(vocab) * laplace_lambda))


def get_each_class_probability(sample, data):
    global classes
    results = {}
    for _class in classes:
        # mle_probability = probability_a_given_b(sample, _class, data)
        mle_probability = probability_a_given_b_M2(sample, _class)
        posterior = posterior_probability_M2(_class, data)
        # posterior = posterior_probability(_class, data)
        # print('Posterior', posterior)
        results[_class] = mle_probability * posterior

    return results


def get_prediction(results):
    # log_list = [math.log(x) for x in list(results.values()) if x > 0]
    max_value = max(list(results.values()))
    index = list(results.values()).index(max_value)
    _class = list(results.keys())[index]
    # print(_class)
    # print(max_value)
    return _class


# ******************** Method-2 for reducing time complexity **************************

def unique_words_N_class_count():
    global _class
    global all_classes_sum
    global all_words_count
    global transformed_df
    for _class in classes:
        data_frame = transformed_df[transformed_df[last_col_name] == _class]
        print(data_frame)
        data_frame = data_frame.drop(last_col_name, axis=1)
        df_sum = data_frame.sum(axis=0).to_numpy()
        all_words_count[_class] = df_sum
        all_classes_sum[_class] = sum(df_sum)
        print('Class', _class, 'Sum', sum(df_sum))


def probability_a_given_b_M2(sample, _class):
    global all_classes_sum
    global all_unique_words
    mle_probability = 1.0
    for word in sample.split():
        count = get_word_count(_class, word)
        n = all_classes_sum[_class]
        # print(count, n, len(all_unique_words))
        prob = probability(count, n, all_unique_words)
        # print('Prob', prob)
        mle_probability *= prob
        # print(word, '=', count, 'Probability', mle_probability)
    return mle_probability


def posterior_probability_M2(_class, data):
    global all_classes_count
    return all_classes_count[_class] / len(data)


def get_word_count(_class, word):
    global all_unique_words
    global all_words_count
    wd_count = 0
    if word in all_unique_words:
        word_index = all_unique_words.index(word)
        wd_count = all_words_count[_class][word_index]
    return wd_count


# *********************************************************************************

def apply_naive_bayes(data, original_data):
    training_pred = []
    for sample in data:
        results = get_each_class_probability(sample, original_data)
        print(results)
        prediction = get_prediction(results)
        training_pred.append(prediction)

    print('\n\n' + '-' * 20)
    print(training_pred)
    return training_pred


if __name__ == '__main__':

    no_of_samples = 10000
    # twitter_df = pd.read_csv('./practice.csv')
    twitter_df = pd.read_csv('./Twitter_Data.csv')
    # twitter_df = twitter_df.sample(frac=1)
    twitter_df = twitter_df[:no_of_samples]
    percent = math.floor(.7 * no_of_samples)
    # print(percent)
    training_data = twitter_df[:percent]
    testing_data = twitter_df[percent:]
    print(len(training_data))
    print(len(testing_data))

    actual_training_data = training_data['clean_text'].to_numpy()
    actual_testing_data = testing_data['clean_text'].to_numpy()
    training_data_output = training_data['category'].to_numpy()
    testing_data_output = testing_data['category'].to_numpy()

    training_tweets, all_unique_words = data_preprocessing(actual_training_data)
    transformed_samples, transformed_df = my_transformation(training_tweets, all_unique_words)

    # transformed_df = pd.read_csv(r'transformed_data.csv')
    print((transformed_df))

    classes = np.unique(training_data_output)
    # print(classes)

    # transformed_df = pd.DataFrame(transformed_samples)
    transformed_df[transformed_df.shape[1]] = training_data['category']
    # transformed_df.to_csv('transformed_data.csv')
    # exit()
    last_col_name = (transformed_df.shape[1] - 1)
    all_words_count = {}
    all_classes_sum = {}
    all_classes_count = {}
    for _class in classes:
        all_classes_count[_class] = len(transformed_df[transformed_df[last_col_name] == _class])
    unique_words_N_class_count()

    print(all_words_count)
    print(all_classes_sum)
    print(all_classes_count)

    training_predictions = apply_naive_bayes(training_tweets, training_tweets)
    print('\n\n' + '-' * 20)
    print(training_data_output)
    training_accuracy = sk.accuracy_score(training_data_output, training_predictions)

    testing_tweets, test_unique_words = data_preprocessing(actual_testing_data)
    # transformed_samples, df = my_transformation(testing_tweets, all_unique_words)
    testing_predictions = apply_naive_bayes(testing_tweets, training_tweets)
    print('\n\n' + '-' * 20)
    print(testing_data_output)
    testing_accuracy = sk.accuracy_score(testing_data_output, testing_predictions)
    print('Training Accuracy:', training_accuracy)
    print('Testing Accuracy:', testing_accuracy)


    # --> for all data
    # training_tweets, all_unique_words = data_preprocessing(twitter_df['clean_text'].to_numpy())
    # transformed_samples, transformed_df = my_transformation(training_tweets, all_unique_words)
    # transformed_df[transformed_df.shape[1]] = training_data['category']
    # transformed_df.to_csv('transformed_data.csv')
    #
    # transformed_samples = pd.read_csv(r'transformed_data.csv')
    # training_trans_samples = transformed_samples[:percent]
    #
    # print('Training Accuracy:', training_accuracy)
    # print('Testing Accuracy:', testing_accuracy)


    # --------------------------------------------------------------------------------------

    # data = ['I loved the movie',
    #         'I hated the movie',
    #         'a great movie good movie',
    #         'poor acting',
    #         'great acting a good movie']
    # output_result = [1, -1, 1, -1, 1]
    # all_unique_words = find_unique_words(data)
    # print(all_unique_words)
    #
    # transformed_samples = np.array(my_transformation(data, all_unique_words))
    # print(transformed_samples)
    # classes = np.unique(output_result)
    # transformed_df = pd.DataFrame(transformed_samples)
    # transformed_df[transformed_df.shape[1]] = output_result
    #
    # transformed_df.to_csv('transformed_data.csv')
    #
    # last_col_name = (transformed_df.shape[1] - 1)
    # all_words_count = {}
    # all_classes_sum = {}
    # all_classes_count = {}
    # for _class in classes:
    #     all_classes_count[_class] = len(transformed_df[transformed_df[last_col_name] == _class])
    # unique_words_N_class_count()
    #
    # print(all_words_count)
    # print(all_classes_sum)
    # print(all_classes_count)
    # # print(transformed_df[transformed_df[last_col_name == 1]].drop(last_col_name, axis=1))
    # # print((transformed_df[transformed_df[transformed_df.shape[1] - 1] == 1]).sum(axis=0).to_numpy())
    # # sample = 'I hated the poor acting'
    #
    # re = get_each_class_probability('I hated the poor acting', data)
    # print(re)
    # prediction = get_prediction(re)
    # print(prediction)
