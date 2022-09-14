import pandas as pd
import numpy as np

from collections import Counter
from scipy.sparse import csr_matrix

from nltk.stem import PorterStemmer

word_stemmer = PorterStemmer()

# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
# stopWords = set(stopwords.words('english'))
# print(stopWords)

tweets = np.array(["Doubt thou the, stars are fire",
                   "Doubt Truth to be a liar",
                   "But never doubt I love.",
                   "I love watching this movie",
                   "I know it's her car."])
# twitter_df = pd.read_csv('./data.csv')
# tweets = twitter_df.head(20)['clean_text'].to_numpy()
# print(len(tweets))
#
# tweets = twitter_df['clean_text']
# twitter_df['length'] = twitter_df['clean_text'].str.len()
# positive = twitter_df[twitter_df['category'] == 1]
# negative = twitter_df[twitter_df['category'] == -1]
# neutral = twitter_df[twitter_df['category'] == 0]

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
        # remove special characters and digits
        line = "".join(ch for ch in line if ch.isalpha() or ch == ' ')

        # remove stops words
        line = " ".join(word for word in line.split() if word not in list(stop_words))

        # apply stemming
        # line = " ".join(word_stemmer.stem(word) for word in line.split())

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


def apply_builtin_stemming(data):
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


vowels = np.array(['a', 'e', 'i', 'o', 'u'])


# word = 'sing'


def step1a(word):
    new_word = ''
    if 'sses' in word[-4:]:
        new_word = word[0:-2]
    elif 'ies' in word[-3:]:
        new_word = word[0:-2]
    elif 'ss' in word[-2:]:
        new_word = word
    elif 's' in word[-1:]:
        new_word = word[0:-1]

    return new_word


def step1b(word):
    new_word = word
    if 'eed' in word[-3:]:
        pass
    elif 'ed' in word[-2:]:
        res = set(True for v in vowels if v in word[0:-2])
        if True in res:
            new_word = word[0:-2]
    elif 'ing' in word[-3:]:
        res = set(True for v in vowels if v in word[0:-3])
        if True in res:
            new_word = word[0:-3]

    return new_word


def step1b_b(word):
    new_word = word
    if 'at' in word[-2:] or 'bl' in word[-2:] or 'iz' in word[-2:]:
        new_word += 'e'
    elif ((word[-2:-1] not in vowels and word[-1:] not in vowels) and
          not (word[-1:] == 'l' or word[-1:] == 's') or word[-1:] == 'z'):
        new_word = word[0:-1]

    return new_word

def step1c(word):
    new_word = word
    is_contain_vowel = set(True for v in vowels if v in word[0:-1])
    if is_contain_vowel and word[-1:] == 'y':
        new_word = word[0:-1] + 'i'

    return new_word


print(step1b_b('hopp'))
print(step1b_b('tann'))
print(step1b_b('fall'))
print(step1b_b('hiss'))
print(step1b_b('fizz'))

print(step1c('happy'))
print(step1c('sky'))

# print(step1b('motoring'))




def my_stemmer(data):
    for index, line in enumerate(data):
        new_line = ''
        for word in line.split():
            word = step1a(word)
            word = step1b(word)
            word = step1b_b(word)
            word = step1c(word)


#
# word = 'caresses'
# print(word[0:-2])
# word = 'ponies'
# print(word[-3:], word[0:-2])
# word = 'caress'
# print(word[-2:], word)
# word = 'cats'
# print(word[-1:], word[0:-1])
#

# print('\n\n Before=>', tweets)

# tweets = remove_punctuation_specChar_digit(tweets)
# tweets = remove_stop_words(tweets)
# tweets = apply_builtin_stemming(tweets)
# all_unique_words = find_unique_words(tweets)

# tweets, all_unique_words = data_preprocessing(tweets)

# print(all_unique_words)
# print('\n\n After=>', tweets)

# print(word_stemmer.stem('movies'))


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
