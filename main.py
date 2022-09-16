import math

import pandas as pd
import numpy as np

from collections import Counter
from scipy.sparse import csr_matrix

# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
# stopWords = set(stopwords.words('english'))
# print(stopWords)

from stemmer import my_line_stemmer

tweets = np.array(["Doubt thou the, stars are fire",
                   "Doubt Truth to be a liar",
                   "But never doubt I love.",
                   "I love love watching this movie",
                   "I know it's her car."])

# tweets = twitter_df['clean_text'].tolist()


# stop_words = {'i', 'me', 'to', 'my', 'myself', 'we', 'our', 'ours', 'your', 'you', 'yourself', 'ourselves',
#               'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'herself',
#               'it', 'its', 'hers', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
#               'which', 'who', 'whom', 'this', 'a', 'be', 'the', 'are'}

stop_words = {'under', 'very', 'shan', 'have', 'wouldn', 'because', 'as', "isn't", "didn't", 'over', 'me', 'into',
              'after', "needn't", "shouldn't", 'she', 'him', 'about', 'these', 'when', 'so', 'hers', 'being', "doesn't",
              'her', 'how', 'too', 'their', 'once', 'at', 'm', 'whom', 'through', 'it', 'does', 'was', 'this', 'than',
              'were', 'off', 's', 'while', "you're", 'by', 'a', 'those', 'both', 'not', 'he', 'having', 'here', 've',
              'y', 'i', 'ain', 'who', "couldn't", 'few', 'other', 'only', 'should', "she's", 'for', 'isn', "wasn't",
              'o', 'such', 'yourself', 'himself', 'ma', 'to', "won't", 'same', 'will', 'do', 'needn', 'that', 'on',
              'down', 're', "should've", 'hasn', 'doing', 'been', 'nor', 'didn', "haven't", 'weren', 'don', 'had',
              'from', 'they', 'ours', 'which', 'doesn', 'theirs', 'is', 'his', "mustn't", 'or', 'again', 'yourselves',
              'did', 'won', "it's", 'aren', 'd', 'with', 'myself', 'our', 'what', 'has', 'why', "aren't", 'of', 'all',
              'any', 'can', 'out', 'haven', 'there', 'between', 'itself', 'yours', 'each', 'no', 'where', "mightn't",
              "hasn't", 'above', 'shouldn', 'herself', 'are', 'am', "you'll", 'themselves', 'we', "weren't", 'you',
              'my', 'the', "you'd", "you've", 'during', 'up', 'its', 'mustn', "that'll", 'in', 'more', 'll', "hadn't",
              'wasn', 'hadn', 'mightn', 'now', 'an', 'couldn', 'most', 'before', 'further', 'them', 'until', 'and',
              'ourselves', "wouldn't", 'just', 'if', 'then', 'below', 'against', 'own', "don't", 'be', 'some', "shan't",
              'your', 't', 'but'}


# special_characters = set('!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~')

def data_preprocessing(list_of_lines):
    unique_words = set()
    for index, line in enumerate(list_of_lines):
        line = line.lower()
        # print('-> ', line)
        # remove special characters and digits
        line = "".join(ch for ch in line if ch.isalpha() or ch == ' ')
        # print('\nSpecial Char:\n', line)

        # remove stops words
        line = " ".join(word for word in line.split() if word not in list(stop_words))
        # print('\nRemove Stop:\n', line)
        # apply stemming
        # line = " ".join(word_stemmer.stem(word) for word in line.split())
        line = my_line_stemmer(line)
        # print('\nStemming:\n', line)

        list_of_lines[index] = line
        # print('\nEnd:\n', line)

        # adding unique words from all lines
        [unique_words.add(word) for word in line.split() if not {word}.issubset(stop_words)]

    return list_of_lines, sorted(list(unique_words))


def remove_punctuation_specChar_digit(data):
    for index, tweet in enumerate(data):
        tweet = tweet.lower()
        tweet = "".join(ch for ch in tweet if ch.isalpha() or ch == ' ')
        data[index] = tweet

    return data


def remove_stop_words(data):
    for index, line in enumerate(data):
        data[index] = " ".join(word for word in line.split() if word not in list(stop_words))

    return data


def find_unique_words(data):
    unique_words = set()
    for line in data:
        for word in line.split():
            # if not {word.lower()}.issubset(stop_words):
            unique_words.add(word)

    return sorted(list(unique_words))


def unique_words_count_dict(unique_word):
    unique_word_dict = {}
    for word in unique_word:
        unique_word_dict[word] = 0
    return unique_word_dict


def transformation(words, unique_words):
    vocab = unique_words_count_dict(unique_words)

    row, col, val = [], [], []
    for idx, word in enumerate(words):

        countWords = dict(Counter(word.lower().split(' ')))

        for w, count in countWords.items():
            if len(w) > 2:
                col_index = vocab.get(w)
                if col_index >= 0:
                    row.append(idx)
                    col.append(col_index)
                    val.append(count)
    return csr_matrix((val, (row, col)), shape=(len(words), len(vocab)))


def my_transformation(tweets, unique_word):
    transformed_array = []
    u_words_dict = unique_words_count_dict(unique_word)

    for line in tweets:
        for u_word in unique_word:
            count = 0
            if u_word in line.split():
                # print(u_word)
                count = sum([1 for word in line.split() if word == u_word])
                # count =[1 for word in line.split() if word == u_word]
                # print(count)
            u_words_dict[u_word] = count

        transformed_array.append(list(u_words_dict.values()))

    return transformed_array


# print('\n\n Before=>', tweets)

# tweets = remove_punctuation_specChar_digit(tweets)
# tweets = remove_stop_words(tweets)
# tweets = apply_builtin_stemming(tweets)
# all_unique_words = find_unique_words(tweets)
twitter_df = pd.read_csv('./data.csv')

# twitter_df = twitter_df.sample(frac=1)
no_of_samples = 20
twitter_df = twitter_df[:no_of_samples]

positive_tweets = twitter_df[twitter_df['category'] == 1]
negative_tweets = twitter_df[twitter_df['category'] == -1]
neutral_tweets = twitter_df[twitter_df['category'] == 0]

percent = math.floor(.7 * no_of_samples)
print(percent)
training_data = twitter_df[:percent]
testing_data = twitter_df[percent:]
print('positive', len(positive_tweets))
print('negative', len(negative_tweets))
print('neutral', len(neutral_tweets))
print('training', len(training_data))
print('testing', len(testing_data))
print('training p', len(training_data[training_data['category'] == 1]))
print('training n', len(training_data[training_data['category'] == -1]))
print('training nt', len(training_data[training_data['category'] == 0]))
print('testing p', len(testing_data[testing_data['category'] == 1]))
print('testing n', len(testing_data[testing_data['category'] == -1]))
print('testing nt', len(testing_data[testing_data['category'] == 0]))

tweets = training_data['clean_text'].to_numpy()
output_result = training_data['category'].to_numpy()

tweets, all_unique_words = data_preprocessing(tweets)
# print(all_unique_words)
transformed_samples = np.array(my_transformation(tweets, all_unique_words))

classes = np.unique(output_result)
print(classes)
# def count_words(sample, counter):
# for word in sample.split():
indices_N_counter_dict = {}
for _class in classes:
    indices_N_counter_dict[_class] = {'indices': [], 'words_count': 0}
# positive_data_dict = {'indices': [], 'words_count': 0}
# negative_data_dict = {'indices': [], 'words_count': 0}
# neutral_data_dict = {'indices': [], 'words_count': 0}

data = ['I loved the movie',
        'I hated the movie',
        'a great movie good movie',
        'poor acting',
        'great acting a good movie']
output_result = [1, -1, 1, -1, 1]
all_unique_words = find_unique_words(data)
print(all_unique_words)

transformed_samples = np.array(my_transformation(data, all_unique_words))
print(transformed_samples)


def append_index_N_count(data_dict, transformed_data, i):
    data_dict['indices'].append(i)
    data_dict['words_count'] += sum(transformed_data[i])


for index, value in enumerate(output_result):
    if value == 1 and value in classes:
        append_index_N_count(indices_N_counter_dict[value], transformed_samples, index)
    elif value == -1:
        append_index_N_count(indices_N_counter_dict[value], transformed_samples, index)
    elif value == 0:
        append_index_N_count(indices_N_counter_dict[value], transformed_samples, index)
# print(positive_data_dict)
# print(negative_data_dict)
# print(neutral_data_dict)

print(indices_N_counter_dict)


def get_case(case):
    if case == '+':
        return 1
    elif case == '-':
        return -1
    elif case == 'o':
        return 0


#  let n be the number of words in the _class case
#  word_k the number of times word k occur
def probability(word_k, n, vocab, laplace_lambda=1):
    return (word_k + laplace_lambda) / (n + (len(vocab) * laplace_lambda))


def get_word_count_in_class(data, word, indices):
    count = 0
    for _index in indices:
        line = data[_index]
        count += sum([1 for wrd in line.split() if wrd == word])
    return count


def probability_a_given_b(sample, _class, data):
    global indices_N_counter_dict
    global all_unique_words
    # print(sample)
    mle_probability = 1.0
    for word in sample.split():
        count = get_word_count_in_class(data, word, indices_N_counter_dict[_class]['indices'])
        prob = probability(count, indices_N_counter_dict[_class]['words_count'], all_unique_words)
        mle_probability *= prob
        # print(word, '=', count, 'Probability', pro)
    return mle_probability


def posterior_probability(_class, data):
    global indices_N_counter_dict
    return indices_N_counter_dict[_class]['words_count'] / len(data)


def get_each_class_probability(sample, data):
    results = {}
    for _class in classes:
        mle_probability = probability_a_given_b(sample, _class, data)
        posterior = posterior_probability(_class, data)
        results[_class] = mle_probability * posterior

    return results


def get_prediction(results):
    print(results)
    # log_list = [math.log(x) for x in list(results.values()) if x > 0]
    max_value = max(list(results.values()))
    index = list(results.values()).index(max_value)
    _class = list(results.keys())[index]
    print(_class)
    print(max_value)


def apply_naive_bayes(data):
    training_pred = []
    for sample in data:
        results = get_each_class_probability(sample, data)
        prediction = get_prediction(results)


re = get_each_class_probability('I hated the poor acting', data)
prediction = get_prediction(re)
# for sample in transformed_matrix:
#     print(sample)


'''
['car', 'doubt', 'fire', 'know', 'liar', 'love', 'movie', 'never', 'star', 'thou', 'truth', 'watche']
[[0 1 1 0 0 0 0 0 1 1 0 0]
 [0 1 0 0 1 0 0 0 0 0 1 0]
 [0 1 0 0 0 1 0 1 0 0 0 0]
 [0 0 0 0 0 1 1 0 0 0 0 1]
 [1 0 0 1 0 0 0 0 0 0 0 0]]


['car', 'doubt', 'fire', 'know', 'liar', 'love', 'movie', 'never', 'star', 'thou', 'truth', 'watche']
[0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
[0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 1]
[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

'''
