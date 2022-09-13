import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
# from wordcloud import wordCloud
from wordcloud.wordcloud import WordCloud
import string
from collections import Counter
from scipy.sparse import csr_matrix

# twitter_df = pd.read_csv('./data.csv')
#
# tweets = twitter_df['clean_text']
# twitter_df['length'] = twitter_df['clean_text'].str.len()
# positive = twitter_df[twitter_df['category'] == 1]
# negative = twitter_df[twitter_df['category'] == -1]
# neutral = twitter_df[twitter_df['category'] == 0]

# tweets = twitter_df['clean_text'].tolist()

tweetsInStr = np.array(["Doubt thou the, stars are fire",
                        "Doubt Truth to be a liar",
                        "But never doubt I love.",
                        "I love watching this movie",
                        "I know it's her car."])

stop_words = {'i', 'me', 'to', 'my', 'myself', 'we', 'our', 'ours', 'your', 'you', 'yourself', 'ourselves',
              'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'herself',
              'it', 'its', 'hers', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'a', 'be', 'the', 'are'}

# special_characters = set('!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~')
unique = set()


def remove_punctuation_specChar_digit():
    for index, tweet in enumerate(tweetsInStr):
        tweet = tweet.lower()
        tweet = "".join(ch for ch in tweet if ch.isalpha() or ch == ' ')
        tweetsInStr[index] = tweet
        # [unique.add(word) for word in tweet.split() if not {word}.issubset(stop_words)]
        # print(stop_words.intersection(set(tweet.split())))
    print(tweetsInStr)


print(unique)


def find_unique_words(data):
    unique_words = set()
    for line in data:
        for word in line.split():
            if not {word.lower()}.issubset(stop_words):
                unique_words.add(word)

    return sorted(list(unique_words))


print(find_unique_words(tweetsInStr))

from nltk.stem import PorterStemmer

# words = tweetsInStr.lower().split()

# print(words)
# for word in words:
#     if word in stopWords:
#         words.remove(word)
# Tokenization /vectorization

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
