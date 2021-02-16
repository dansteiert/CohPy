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

def word_frequency(document_word_freq_dict, df_background_corpus_frequency, document_size, background_corpus_size=1000000, frequency_name="frequency"):
    '''
    Ref: CohMetrix - Grasser2004 - Word Frequency
    Ref: Pitler08 - Vocabulary -> log likelihood of an article: sum C(w) * log(P(w|M)); C(w) count of word; P(w|M) probability of w occuring in M; M is the background Corpus
    https://wortschatz.uni-leipzig.de/en/download Word Freq origin
    Calculates the mean word frequency, by sentences
    :param lemma_by_sent: list, list of list of lemma
    :param word_freq_dict: dict, frequency directory for words in German
    :return: int
    '''

    # freq_in_corpus = []
    # word_familarity = []
    # for k, v in document_word_freq_dict.items():
        # try:
        #     temp_row = df_background_corpus_frequency.query(expr="index == '%s'" % k)
        # except:
        #     continue
        # if temp_row.shape[0] > 0:
        #     try:
        #         freq_in_corpus.append(np.log10(float(temp_row.loc[k, "frequency"]) / background_corpus_size))
        #         word_familarity.append(np.log10(v/document_size * (float(temp_row.loc[k, "frequency"]) / background_corpus_size)))
        #     except:
        #         freq_in_corpus.append(min_freq)
        #         word_familarity.append(v/document_size * min_freq)
        # else:
        #     freq_in_corpus.append(min_freq)
        #     word_familarity.append(v / document_size * min_freq)


    freq_in_corpus = {k: float(df_background_corpus_frequency.get(k, {}).get(frequency_name, 1)) / background_corpus_size for k, v in document_word_freq_dict.items()}
    freq_in_document = {k: (v / document_size) * (float(freq_in_corpus.get(k, 1)) / background_corpus_size) for k, v in document_word_freq_dict.items()}

    corpus_freq = np.array([np.log10(v) for v in freq_in_corpus.values()])
    text_freq = np.array([np.log10(v) for v in freq_in_document.values()])
    corr_matrix = np.corrcoef(x=corpus_freq, y=text_freq)
    unique_words = [True for v in document_word_freq_dict.values() if v == 1]
    # print(freq_in_corpus)
    # print(freq_in_document)
    # print(corpus_freq)
    # print(text_freq)
    # print(corr_matrix)
    if document_size > 0:
        return mean_of_list(corpus_freq), corr_matrix[0, 1], len(unique_words)/len(freq_in_corpus)
    else:
        return mean_of_list(corpus_freq), corr_matrix[0, 1], len(unique_words)