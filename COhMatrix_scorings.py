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



# TODO: Excluded are Noun Phrases (lack of implementation/knowledge)
def co_reference_matrix(document_tag, document_lemma, accept_tags=[], accept_tags_start_with=["N", "P"],
                        exclude_tags=["PTK"],
                        exclude_tags_start_with=[], punctuation_tag_group=["$."]):
    # TODO: redo/rethink

    word_dict = {}
    sentence_count = 0
    for l, t in zip(document_lemma, document_tag):
        if check_tags(tag=t, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                      exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with):
            temp_dict = word_dict.get(l, [])
            temp_dict.append(sentence_count)
            word_dict[l] = temp_dict
        elif t in punctuation_tag_group:
            sentence_count += 1
    if sentence_count < 2:
        return None

    # word_list = word_dict.keys()
    co_reference_exists = np.zeros(shape=(sentence_count, sentence_count))
    co_reference_dist = np.zeros(shape=(sentence_count, sentence_count))
    for val in word_dict.values():
        for index, i in enumerate(val):
            try:
                for index_j, j in enumerate(val[index + 1:]):
                    co_reference_exists[i, j] = 1
                    co_reference_dist[i, j] = 1 / abs(i - j)
            except:
                pass
    local_corefererence_cohesion = (1 / (sentence_count - 1)) * sum(
        [i[index + 1] for index, i in enumerate(co_reference_exists) if index + 1 < len(co_reference_exists)])
    global_corefererence_cohesion = (1 / (sentence_count * ((sentence_count - 1) / 2))) * np.sum(co_reference_exists)
    co_reference_dist_sum = np.sum(co_reference_dist) * 1 / sentence_count
    # print(local_corefererence_cohesion, global_corefererence_cohesion, co_reference_dist_sum)
    # print(co_reference_dist)
    # print(co_reference_exists)
    return (local_corefererence_cohesion, global_corefererence_cohesion, co_reference_dist_sum)
