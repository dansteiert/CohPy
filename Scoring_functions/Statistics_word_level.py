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
    Heuristical approximation
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


def word_frequency(document_word_freq_dict, document_size, df_background_corpus_frequency, background_corpus_size=1000000, frequency_name="frequency"):
    '''
    Ref: CohMetrix - Grasser2004 - Word Frequency
    Ref: Pitler08 - Vocabulary -> log likelihood of an article: sum C(w) * log(P(w|M)); C(w) count of word; P(w|M) probability of w occuring in M; M is the background Corpus
    https://wortschatz.uni-leipzig.de/en/download Word Freq origin
    Calculates the mean word frequency, correlation of background corpus to document and the incidence of unique words
    
    :param document_word_freq_dict: dict, {key=lemma, value=absolut count}
    :param document_size: int, # sentences in document
    :param df_background_corpus_frequency: dict, {key=lemma, value={key="column label", value=absolut count in background corpus}}
    :param background_corpus_size: int, # sentences in background corpus
    :param frequency_name: name of frequency column in background data file
    :return: float mean of log 10 transformd (how often a word occurrce in the background corpus);
    float: correlation of background corpus count to document count of lemma;
    flaot: incidence of unique words
    '''

    freq_in_corpus = {k: float(df_background_corpus_frequency.get(k, {}).get(frequency_name, 1)) / background_corpus_size for k, v in document_word_freq_dict.items()}
    freq_in_document = {k: (v / document_size) * (float(freq_in_corpus.get(k, 1)) / background_corpus_size) for k, v in document_word_freq_dict.items()}

    corpus_freq = np.array([np.log10(v) for v in freq_in_corpus.values()])
    text_freq = np.array([np.log10(v) for v in freq_in_document.values()])
    corr_matrix = np.corrcoef(x=corpus_freq, y=text_freq)
    unique_words = [True for v in document_word_freq_dict.values() if v == 1]

    try:
        return mean_of_list(corpus_freq), corr_matrix[0, 1], len(unique_words)/1000
    except:
        return None, None, None