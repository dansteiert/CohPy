from Helper_functions import *
from w2v_model import sentence_similarity
import numpy as np
def tag_overlap(sent_a_tags, sent_a_lemma, sent_b_tags, sent_b_lemma, accept_tags=[], accept_tags_start_with=["N"],
                exclude_tags=[], exclude_tags_start_with=[]):
  '''
  :param sent_a_tags: tag list of sentance/ pargraph/.. a
  :param sent_a_lemma: lemma list of sentance/ pargraph/.. a
  :param sent_b_tags: tag list of sentance/ pargraph/.. b
  :param sent_b_lemma: lemma list of sentance/ pargraph/.. b
  :return: number of overlapping lemma
  '''
  lemma_set_sent_a = set(search_tag_set(aggregate=sent_a_lemma, tags=sent_a_tags, accept_tags=accept_tags,
                                        accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                                        exclude_tags_start_with=exclude_tags_start_with))
  lemma_set_sent_b = set(search_tag_set(aggregate=sent_b_lemma, tags=sent_b_tags, accept_tags=accept_tags,
                                        accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                                        exclude_tags_start_with=exclude_tags_start_with))

  overlapping_lemma = lemma_set_sent_a.intersect(lemma_set_sent_b)
  return len(overlapping_lemma)


def word_repetition(document_lemma, document_tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=["ART"],
                    exclude_tags_start_with=["$"]):
  # is it important how often a word occured multiple times
  lemma_list = search_tag_set(aggregate=document_lemma, tags=document_tags, accept_tags=accept_tags,
                            accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                            exclude_tags_start_with=exclude_tags_start_with)
  count_dict = to_count_dict(aggregate_list=lemma_list)

  repeated_words = [k for k, v in count_dict.items() if v > 1]
  return len(repeated_words)


def overlap_matrix(lemma_by_sentence, tags_by_sentence, accept_tags=[], accept_tags_start_with=[],
                   exclude_tags=[], exclude_tags_start_with=[]):
    m = np.zeros((len(lemma_by_sentence), len(lemma_by_sentence)))
    for index_a, (l_a, t_a) in enumerate(zip(lemma_by_sentence, tags_by_sentence)):
        for index_b, (l_b, t_b) in enumerate(zip(lemma_by_sentence[index_a + 1:], tags_by_sentence[index_a + 1:])):
            m[index_a, index_a + 1 + index_b] = tag_overlap(sent_a_lemma=l_a, sent_a_tags=t_a, sent_b_lemma=l_b, sent_b_tags=t_b, accept_tags=accept_tags,
                        accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                        exclude_tags_start_with=exclude_tags_start_with)
    return m

def overlap_matrix_sentiment(w2v_model, lemma_by_sentence, tags_by_sentence, accept_tags=[], accept_tags_start_with=[],
                   exclude_tags=[], exclude_tags_start_with=[]):
    m = np.zeros((len(lemma_by_sentence), len(lemma_by_sentence)))
    hit_elements = 0
    searched_elements = 0
    for index_a, (l_a, t_a) in enumerate(zip(lemma_by_sentence, tags_by_sentence)):
        for index_b, (l_b, t_b) in enumerate(zip(lemma_by_sentence[index_a + 1:], tags_by_sentence[index_a + 1:])):
            v_temp, h_temp, s_temp = sentence_similarity(w2v=w2v_model, sent_a_lemma=l_a, sent_a_tags=t_a,
                                                         sent_b_lemma=l_b, sent_b_tags=t_b, accept_tags=accept_tags,
                                                         accept_tags_start_with=accept_tags_start_with,
                                                         exclude_tags=exclude_tags,
                                                         exclude_tags_start_with=exclude_tags_start_with)
            m[index_a, index_a + 1 + index_b] = v_temp
            hit_elements += h_temp
            searched_elements += s_temp
    hitrate = hit_elements/searched_elements
    return m, hitrate