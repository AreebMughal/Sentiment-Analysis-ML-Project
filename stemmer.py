from nltk.stem import PorterStemmer
import numpy as np

word_stemmer = PorterStemmer()

vowels = np.array(['a', 'e', 'i', 'o', 'u'])


def apply_builtin_stemming(data):
    for index, line in enumerate(data):
        line = " ".join(word_stemmer.stem(word) for word in line.split())
        data[index] = line

    return data


# word = 'sing'

def is_consonant(word, i):
    if word[i] in vowels:
        return False
    if word[i] == 'y':
        if i != 0:
            return not (word[i - 1] not in vowels)
        else:
            return True
    return True


def end_with_double_consonant(word):
    return len(word) >= 2 and word[-1] == word[-2] and word[-1] != 'y'


def end_with_cvc_starO(word):
    sequence = ''
    for i in range(len(word[-3:])):
        if is_consonant(word, i):
            sequence += 'c'
        else:
            sequence += 'v'

    if (len(sequence) >= 3 and (sequence[-3:] == 'cvc') and
            not (word[-1] == 'w' or word[-1] == 'x' or word[-1] == 'y')):
        return True
    else:
        return False


# print('fryy', is_consonant('fryy', 3))

def vc_measure(word):
    sequence = ''
    for i in range(len(word)):
        if is_consonant(word.lower(), i):
            sequence += 'c'
        else:
            sequence += 'v'
    # print(sequence)
    return sequence.count('vc')


# ar = ['TR', 'EE', 'TREE', 'Y', 'BY', 'TROUBLE', 'OATS', 'TREES', 'IVY', 'TROUBLES', 'PRIVATE', 'OATEN', 'ORRERY']
# for word in ar:
#     print(word, '->', vc_measure(word))

def step1a(word):
    new_word = word
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
    # (m>0) EED	-> EE
    if 'eed' in word[-3:]:
        m = vc_measure(word[0:3])
        if m > 0:
            new_word = word[0:-1]
        return new_word

    flag = False
    # (*v*) ED ->
    if 'ed' in word[-2:]:
        res = set(True for v in vowels if v in word[0:-2])
        if True in res:
            new_word = word[0:-2]
            flag = True
    #  (*v*) ING ->
    elif 'ing' in word[-3:]:
        res = set(True for v in vowels if v in word[0:-3])
        if True in res:
            new_word = word[0:-3]
            flag = True

    if flag:
        new_word = step1b_b(new_word)

    return new_word


def step1b_b(word):
    new_word = word
    # AT or BL or IZ -> *e
    if 'at' in word[-2:] or 'bl' in word[-2:] or 'iz' in word[-2:]:
        new_word += 'e'
    # (*d and not (*L or *S or *Z)) -> single letter
    elif (end_with_double_consonant(word) and
          not (word[-1:] == 'l' or word[-1:] == 's') or word[-1:] == 'z'):
        new_word = word[0:-1]
    else:
        m = vc_measure(word)
        if m == 1 and end_with_cvc_starO(word):
            new_word = word + 'e'
    return new_word


def step1c(word):
    new_word = word
    is_contain_vowel = set(True for v in vowels if v in word[0:-1])
    if is_contain_vowel and word[-1:] == 'y':
        new_word = word[0:-1] + 'i'

    return new_word


def step2_3_4_general_condition_replace(step, word, suffix, replace_with):
    if step > 1:
        m_range = 0
        if step == 4:
            m_range = 1
        prefix_len = abs(len(word) - len(suffix))
        word_without_suffix = word[:prefix_len]
        m = vc_measure(word_without_suffix)
        # print(word, word_without_suffix, suffix, m, m > 0 and word.endswith(suffix))
        if step == 4 and suffix == 'ion':
            if len(word) > 1:
                if ((m > m_range and (word[-1] == 's' or word[-1] == 't'))
                        and (word.endswith(suffix))):
                    return word.replace(suffix, replace_with)
        else:
            if m > m_range and word.endswith(suffix):
                return word.replace(suffix, replace_with)

    return word


def step2(word):
    new_word = word
    # if is_step2_general_condition(word, 'ational', 'ate'):
    #     new_word = word.replace('ational', 'ate')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'ational', 'ate')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'tional', 'tion')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'enci', 'ence')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'anci', 'ance')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'izer', 'ize')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'abli', 'able')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'alli', 'al')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'entli', 'ent')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'eli', 'e')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'ousli', 'ous')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'ization', 'ize')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'ation', 'ate')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'ator', 'ate')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'alism', 'al')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'iveness', 'ive')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'fulness', 'ful')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'ousness', 'ous')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'aliti', 'al')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'iviti', 'ive')
    new_word = step2_3_4_general_condition_replace(2, new_word, 'biliti', 'ble')

    return new_word

# def step3_general_condition_replace(word, suffix, replace_with):
#     prefix_len = abs(len(word) - len(suffix))
#     word_without_suffix = word[:prefix_len]
#     m = vc_measure(word_without_suffix)
#     # print(word, word_without_suffix, suffix, m, m > 0 and word.endswith(suffix))
#     if m > 0 and word.endswith(suffix):
#         return word.replace(suffix, replace_with)
#     return word

def step3(word):
    new_word = word
    new_word = step2_3_4_general_condition_replace(3, new_word, 'biliti', 'ble')
    new_word = step2_3_4_general_condition_replace(3, new_word, 'icate', 'ic')
    new_word = step2_3_4_general_condition_replace(3, new_word, 'atvie', '')
    new_word = step2_3_4_general_condition_replace(3, new_word, 'alize', 'al')
    new_word = step2_3_4_general_condition_replace(3, new_word, 'iciti', 'ic')
    new_word = step2_3_4_general_condition_replace(3, new_word, 'ical', 'ic')
    new_word = step2_3_4_general_condition_replace(3, new_word, 'ful', '')

    return new_word


def step4(word):
    new_word = word
    new_word = step2_3_4_general_condition_replace(4, new_word, "al", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ance", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ence", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "er", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ic", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "able", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ible", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ant", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ement", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ment", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ent", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ion", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ou", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ism", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ate", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "iti", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ous", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ive", "")
    new_word = step2_3_4_general_condition_replace(4, new_word, "ize", "")

    return new_word


def step5a(word):
    new_word = word
    if len(word) > 1:
        if word[-1] == 'e':
            m = vc_measure(word[0:-1])
            if m > 1:
                new_word = word[0:-1]
            if m == 1 and not end_with_cvc_starO(word[0:-1]):
                new_word = word[0:-1]
    return new_word


def step5b(word):
    new_word = word
    m = vc_measure(word)
    if len(word) > 1:
        if m > 1 and end_with_double_consonant(word) and word[-1] == 'l':
            new_word = word[0:-1]

    return new_word


def my_stemmer(data):
    for index, line in enumerate(data):
        new_line = ''
        for word in line.split():
            word = step1a(word)
            word = step1b(word)
            word = step1c(word)
            word = step3(word)
            word = step4(word)
            word = step5a(word)
            word = step5b(word)
            new_line += word + ' '

        data[index] = new_line
    return data


def my_line_stemmer(line):
    new_line = ''
    for word in line.split():
        word = step1a(word)
        word = step1b(word)
        word = step1c(word)
        word = step3(word)
        word = step4(word)
        word = step5a(word)
        word = step5b(word)
        # print(word)
        new_line += word + ' '

    return new_line

#
# print('\n Step - 2')
# print(step2('relational'))
# print(step2('conditional'))
# print(step2('rational'))
# print(step2('valenci'))
# print(step2('hesitanci'))
# print(step2('digitizer'))
# print(step2('conformabli'))
# print(step2('radicalli'))
# print(step2('differentli'))
# print(step2('vileli'))
# print(step2('analogousli'))
# print(step2('vietnamization'))
# print(step2('predication'))
# print(step2('operator'))
# print(step2('feudalism'))
# print(step2('decisiveness'))
# print(step2('hopefulness'))
# print(step2('callousness'))
# print(step2('formaliti'))
# print(step2('sensitiviti'))
# print(step2('sensibiliti'))
#
# print('\n Step - 3')
# print(step3('triplicate'))
# print(step3('formative'))
# print(step3('formalize'))
# print(step3('electriciti'))
# print(step3('electrical'))
# print(step3('hopeful'))
# print(step3('goodness'))
#
# print('\n Step - 4')
# print(step4('revival'))
# print(step4('allowance'))
# print(step4('inference'))
# print(step4('airliner'))
# print(step4('gyroscopic'))
# print(step4('adjustable'))
# print(step4('defensible'))
# print(step4('irritant'))
# print(step4('replacement'))
# print(step4('adjustment'))
# print(step4('dependent	'))
# print(step4('adoption'))
# print(step4('homologou'))
# print(step4('communism'))
# print(step4('activate'))
# print(step4('angulariti'))
# print(step4('homologous'))
# print(step4('effective'))
# print(step4('bowdlerize'))
#
# print('\n Step - 5a')
# print(step5a('probate'))
# print(step5a('rate'))
# print(step5a('cease'))
#
# print('\n Step - 5b')
# print(step5b('controll'))
# print(step5b('roll'))

# print(vc_measure('feed'))

# print(step1b('filing'))
# print(step1b('conflated'))
# print(step1b('tanned'))
# print(step1b('falling'))
# print(step1b_b('hopxp'))


# print(step1b_b('tann'))
# print(step1b_b('fall'))
# print(step1b_b('hiss'))
# print(step1b_b('fizz'))
#
# print(step1c('happy'))
# print(step1c('sky'))

# word = 'caresses'
# print(word[0:-2])
# word = 'ponies'
# print(word[-3:], word[0:-2])
# word = 'caress'
# print(word[-2:], word)
# word = 'cats'
# print(word[-1:], word[0:-1])


# print(step1b('motoring'))
