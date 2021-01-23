import pandas as pd
import os
import numpy as np
from Helper_functions import *




def word_familarity(document_word, familarity_dict):
    # retrieve frequency from a DB of Tests
    count = 0
    for i in document_word:
        temp = familarity_dict.get(i, None)
        if temp is not None:
            count += temp
    count /= len(document_word)
    return count


# TODO: causal cohesion
def casual_cohesion(document):
    # The total list of causal particles comes either from this short list of verbs or from the causal conjunctions,
    # transitional adverbs, and causal connectives. The current metric of causal cohesion, which is a primary measure,
    # is simply a ratio of causal particles (P) to causal verbs (V).

    return None


def Flescher_Reading_Ease(document_words, document_syllables, num_sentences):
    if len(document_words) < 200:
        return None

    ASL = len(document_words) / num_sentences  # ratio #words/#sent
    ASW = sum(document_syllables) / len(document_words)  # # ratio Syllables/Words
    return 206.835 - 1.015 * ASL - 84.6 * ASW


def Flescher_Kincaid_Grade_Level(document_words, document_syllables, num_sentences):
    if len(document_words) < 200:
        return None

    ASL = len(document_words) / num_sentences  # ratio #words/#sent
    ASW = sum(document_syllables) / len(document_words)
    return 0.39 * ASL + 11.8 * ASW - 15.59


## More readability scores: (from Art)
def readability_metrics(text):
    """ Returns a dictionary containing all readability scores for a given text """
    return {
        'EN: flesch_kincaid_ease': flesch_kincaid_ease(text),
        'IT: gulpease': gulpease(text),
        'FR: kandel_moles': kandel_moles(text),
        'EN: flesch_kincaid_grade': flesch_kincaid_grade(text),
        'EN: gunning_fog': gunning_fog(text),
        'EN: coleman_liau': coleman_liau(text),
        'EN: smog': smog(text),
        'EN: ari': ari(text),
        'DE:flesch_kincaid_ease': FKDE(text)
    }

"""
Readability indices: higher scores imply "easier" reading
"""

def flesch_kincaid_ease(text):
    """ http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_Reading_Ease
    Score	School level	Notes
    100.00-90.00 5th grade	Very easy to read. Easily understood by an average 11-year-old student.
    90.0–80.0	6th grade	Easy to read. Conversational English for consumers.
    80.0–70.0	7th grade	Fairly easy to read.
    70.0–60.0	8th & 9th grade	Plain English. Easily understood by 13- to 15-year-old students.
    60.0–50.0	10th to 12th grade	Fairly difficult to read.
    50.0–30.0	College	Difficult to read.
    30.0–0.0 	College graduate Very difficult to read. Best understood by university graduates.
    """
    text = preprocess(text)
    return 206.835 - (1.015 * avg_words_per_sentence(text)) - (84.6 * avg_syllables_per_word(text))

def FKDE(text):
    """ http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_Reading_Ease """
    text = preprocess(text)
    return 180 - avg_words_per_sentence(text) - (58 * avg_syllables_per_word(text))

def douma(text):
    """ Variant of Flesch-Kincaid for Dutch: http://www.cnts.ua.ac.be/papers/2002/Geudens02.pdf """
    text = preprocess(text)
    return 206.84 - (0.33 * avg_words_per_sentence(text)) - (0.77 * avg_syllables_per_word(text))

def kandel_moles(text):
    """ Variant of Flesch-Kincaid for French (citation not easily traceable) """
    text = preprocess(text)
    return 209 - (1.15 * avg_words_per_sentence(text)) - (0.68 * avg_syllables_per_word(text))

def gulpease(text):
    """ https://it.wikipedia.org/wiki/Indice_Gulpease """
    text = preprocess(text)
    return 89.0 + (300.0 * sentence_count(text) - 10.0 * letter_count(text))/(word_count(text))

def fernandez_huerta(text):
    """ Developed for Spanish texts (citation not easily traceable) """
    text = preprocess(text)
    factor = 100.0 / word_count(text)
    return 206.84 - (0.6 * factor * syllable_count(text)) - (1.02 * factor * sentence_count(text))


"""
Grade level estimators: higher scores imply more advanced-level material
"""

def flesch_kincaid_grade(text):
    """ http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch.E2.80.93Kincaid_Grade_Level """
    text = preprocess(text)
    return (0.39 * avg_words_per_sentence(text)) + (11.8 * avg_syllables_per_word(text)) - 15.59

def gunning_fog(text):
    """ http://en.wikipedia.org/wiki/Gunning_Fog_Index """
    text = preprocess(text)
    return 0.4 * (avg_words_per_sentence(text) + percent_three_syllable_words(text, False))

def coleman_liau(text):
    """ http://en.wikipedia.org/wiki/Coleman-Liau_Index """
    text = preprocess(text)
    return  (5.89 * letter_count(text) / word_count(text)) - (0.3 * sentence_count(text) / word_count(text)) - 15.8

def smog(text):
    """ http://en.wikipedia.org/wiki/SMOG_Index """
    text = preprocess(text)
    return 1.043 * sqrt((three_syllable_word_count(text) * (30.0 / sentence_count(text))) + 3.1291)

def ari(text):
    """ http://en.wikipedia.org/wiki/Automated_Readability_Index """
    text = preprocess(text)
    return (4.71 * letter_count(text) / word_count(text)) + (0.5 * word_count(text) / sentence_count(text)) - 21.43

"""
Other indices (not grade levels): higher scores imply "more difficult" reading
"""

def lix(text):
    """ http://en.wikipedia.org/wiki/LIX """
    text = preprocess(text)
    num_words = word_count(text)
    return (100.0 * six_letter_word_count(text) / num_words) + (1.0 * num_words / sentence_count(text))

def rix(text):
    """ More generalized variant of LIX """
    text = preprocess(text)
    return 1.0 * six_letter_word_count(text) / sentence_count(text)



# # TODO: Excluded are Noun Phrases (lack of implementation/knowledge)
# def co_reference_matrix(document_tag, document_lemma, accept_tags=[], accept_tags_start_with=["N", "P"],
#                         exclude_tags=["PTK"],
#                         exclude_tags_start_with=[], punctuation_tag_group=["$."]):
#     # TODO: redo/rethink
#
#     word_dict = {}
#     sentence_count = 0
#     for l, t in zip(document_lemma, document_tag):
#         if check_tags(tag=t, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
#                       exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with):
#             temp_dict = word_dict.get(l, [])
#             temp_dict.append(sentence_count)
#             word_dict[l] = temp_dict
#         elif t in punctuation_tag_group:
#             sentence_count += 1
#     if sentence_count < 2:
#         return None
#
#     # word_list = word_dict.keys()
#     co_reference_exists = np.zeros(shape=(sentence_count, sentence_count))
#     co_reference_dist = np.zeros(shape=(sentence_count, sentence_count))
#     for val in word_dict.values():
#         for index, i in enumerate(val):
#             try:
#                 for index_j, j in enumerate(val[index + 1:]):
#                     co_reference_exists[i, j] = 1
#                     co_reference_dist[i, j] = 1 / abs(i - j)
#             except:
#                 pass
#     local_corefererence_cohesion = (1 / (sentence_count - 1)) * sum(
#         [i[index + 1] for index, i in enumerate(co_reference_exists) if index + 1 < len(co_reference_exists)])
#     global_corefererence_cohesion = (1 / (sentence_count * ((sentence_count - 1) / 2))) * np.sum(co_reference_exists)
#     co_reference_dist_sum = np.sum(co_reference_dist) * 1 / sentence_count
#     # print(local_corefererence_cohesion, global_corefererence_cohesion, co_reference_dist_sum)
#     # print(co_reference_dist)
#     # print(co_reference_exists)
#     return (local_corefererence_cohesion, global_corefererence_cohesion, co_reference_dist_sum)
