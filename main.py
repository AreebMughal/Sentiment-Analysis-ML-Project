from nltk.stem import PorterStemmer

word_stemmer = PorterStemmer()
from collections import Counter
from scipy.sparse import csr_matrix
from stemmer import my_line_stemmer

# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
# stopWords = set(stopwords.words('english'))
# print(stopWords)

stop_words = {
    'under', 'very', 'shan', 'have', 'wouldn', 'because', 'as', "isn't", "didn't", 'over', 'me', 'into',
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
    'your', 't', 'but'
}


def data_preprocessing(list_of_lines):
    unique_words = set()
    for index, line in enumerate(list_of_lines):
        line = str(line)
        if type(line) == str:
            print('-> ', line)
            line = line.lower()
            # remove special characters and digits
            line = "".join(ch for ch in line if ch.isalpha() or ch == ' ')
            # print('\nSpecial Char:\n', line)

            # remove stops words
            line = " ".join(word for word in line.split() if word not in list(stop_words))
            # print('\nRemove Stop:\n', line)
            # apply stemming
            line = " ".join(word_stemmer.stem(word) for word in line.split())
            # line = my_line_stemmer(line)
            # print('\nStemming:\n', line)

            list_of_lines[index] = line
            # print('\nEnd:\n', line)

            # adding unique words from all lines
            [unique_words.add(word) for word in line.split() if not {word}.issubset(stop_words)]
        else:
            print('\n\n' + '*'*20)
            print(index)
            print(line, '\n')
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

