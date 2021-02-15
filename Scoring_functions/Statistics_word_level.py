from Helper.Helper_functions import mean_of_list, to_count_dict
import numpy as np

def word_length(document_word):
    '''
    Ref: Pitler08 - Baseline measures
    :param document: List of tokenized entries
    :return: length for each entry in the list
    '''
    return mean_of_list([len(i) for i in document_word])


def syllable_count(document_words):
    '''
    Ref: Needed for Flescher_Reading_Ease, Flescher_Kincaid_Grade_Level
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

def word_frequency(document_word_freq_dict, background_corpus_word_freq_dict, document_size, background_corpus_size=1000000):
    '''
    Ref: CohMetrix - Grasser2004 - Word Frequency
    Ref: Pitler08 - Vocabulary -> log likelihood of an article: sum C(w) * log(P(w|M)); C(w) count of word; P(w|M) probability of w occuring in M; M is the background Corpus
    https://wortschatz.uni-leipzig.de/en/download Word Freq origin
    Calculates the mean word frequency, by sentences
    :param lemma_by_sent: list, list of list of lemma
    :param word_freq_dict: dict, frequency directory for words in German
    :return: int
    '''
    corp_min_freq = 1/background_corpus_size
    corpus_word_freq = [(background_corpus_word_freq_dict.get(k, corp_min_freq)/background_corpus_size,
                         np.log10((v * (document_word_freq_dict.get(k, corp_min_freq)/background_corpus_size)))) for k, v in document_word_freq_dict.items()]
    corpus_freq = np.array([i[0] for i in corpus_word_freq])
    word_freq = np.array([i[0] for i in corpus_word_freq])
    text_freq = np.array(document_word_freq_dict.values())
    corr_matrix = np.corrcoef(x=corpus_freq, y=text_freq)
    return mean_of_list(word_freq), corr_matrix[0, 1], len(word_freq)/(document_size/1000)
        