import pandas as pd
import numpy as np

from collections import Counter
from scipy.sparse import csr_matrix

from nltk.stem import PorterStemmer

word_stemmer = PorterStemmer()
# twitter_df = pd.read_csv('./data.csv')
#
# tweets = twitter_df['clean_text']
# twitter_df['length'] = twitter_df['clean_text'].str.len()
# positive = twitter_df[twitter_df['category'] == 1]
# negative = twitter_df[twitter_df['category'] == -1]
# neutral = twitter_df[twitter_df['category'] == 0]

# tweets = twitter_df['clean_text'].tolist()

tweets = np.array(["Doubt thou the, stars are fire",
                   "Doubt Truth to be a liar",
                   "But never doubt I love.",
                   "I love watching this movie",
                   "I know it's her car."])

stop_words = {'i', 'me', 'to', 'my', 'myself', 'we', 'our', 'ours', 'your', 'you', 'yourself', 'ourselves',
              'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'herself',
              'it', 'its', 'hers', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'a', 'be', 'the', 'are'}


# special_characters = set('!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~')

def data_preprocessing(list_of_lines):
    unique_words = set()
    for index, line in enumerate(list_of_lines):
        line = line.lower()
        # remove special characters and digits
        line = "".join(ch for ch in line if ch.isalpha() or ch == ' ')

        # remove stops words
        line = " ".join(word for word in line.split() if word not in list(stop_words))

        # apply stemming
        line = " ".join(word_stemmer.stem(word) for word in line.split())

        list_of_lines[index] = line

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


def apply_stemming(data):
    for index, line in enumerate(data):
        line = " ".join(word_stemmer.stem(word) for word in line.split())
        data[index] = line

    return data


def find_unique_words(data):
    unique_words = set()
    for line in data:
        for word in line.split():
            if not {word.lower()}.issubset(stop_words):
                unique_words.add(word)

    return sorted(list(unique_words))


print('\n\n Before=>', tweets)

# tweets = remove_punctuation_specChar_digit(tweets)
# tweets = remove_stop_words(tweets)
# tweets = apply_stemming(tweets)
# all_unique_words = find_unique_words(tweets)

tweets, all_unique_words = data_preprocessing(tweets)

print(all_unique_words)
print('\n\n After=>', tweets)

print(word_stemmer.stem('movies'))



def UniqueWords(data):
    unique = set()
    for words in data:
        for word in words.lower().split(' '):
            if len(word) > 2:
                unique.add(word)
    vocabOfUniqueWords = {}
    uniqueList = list(unique)
    sortList = sorted(uniqueList)
    for index, word in enumerate(sortList):
        vocabOfUniqueWords[word] = index
    return vocabOfUniqueWords


# print(UniqueWords(tweetsInStr))


def transformation(words):
    vocab = UniqueWords(words)

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

# print(transformation(twitter_df['clean_text']).toarray())
# print(transformation(tweetsInStr))
