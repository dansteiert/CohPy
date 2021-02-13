from Helper.Helper_functions import mean_of_list
import numpy as np

def word_length(document_word):
    '''

    :param document: List of tokenized entries
    :return: length for each entry in the list
    '''
    return mean_of_list([len(i) for i in document_word])


def syllable_count(document_words):
    '''

    :param document: List of tokenized entries
    :return: Syllable count for each entry in the list
    '''
    syllable_list = []
    for i in document_words:
        count = 0
        flipper = False
        for j in i.lower():
            if flipper:
                flipper = False
                continue
            if j in "aeiou":
                count += 1
                flipper = True
        if count == 0:
            continue
        syllable_list.append(count)
    return syllable_list

def word_frequency(lemma, word_freq_dict, word_dict_corpus_size=1000000):
    '''
    Ref: CohMetrix - Grasser2004 - Word Frequency
    https://wortschatz.uni-leipzig.de/en/download Word Freq origin
    Calculates the mean word frequency, by sentences
    :param lemma_by_sent: list, list of list of lemma
    :param word_freq_dict: dict, frequency directory for words in German
    :return: int
    '''
    min_freq = 1/word_dict_corpus_size
    word_freq = [word_freq_dict.get(l, min_freq) for l in lemma]
    word_freq = [np.log10(i/word_dict_corpus_size) for i in word_freq]
    return word_freq
        