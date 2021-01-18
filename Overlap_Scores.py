from Helper_functions import *
from w2v_model import sentence_similarity
import numpy as np
import datetime


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
    # print("----------")
    # print(lemma_set_sent_a)
    # print(lemma_set_sent_b)
    overlapping_lemma = lemma_set_sent_a.intersection(lemma_set_sent_b)
    # print(overlapping_lemma)
    return len(overlapping_lemma)



def overlap_matrix(lemma_by_segment, tags_by_segment, accept_tags=[], accept_tags_start_with=[],
                   exclude_tags=[], exclude_tags_start_with=[]):
    v = 0
    for index_a, (l_a, t_a) in enumerate(zip(lemma_by_segment, tags_by_segment)):
        # if (index_a % int(len(lemma_by_segment) * 0.1)) == 0 and index_a != 0:
        #     # print(datetime.datetime.now(), "10% done total segments:", len(lemma_by_segment))
        if index_a + 1 >= len(lemma_by_segment):
            continue
        v += tag_overlap(sent_a_lemma=l_a, sent_a_tags=t_a, sent_b_lemma=lemma_by_segment[index_a + 1],
                         sent_b_tags=tags_by_segment[index_a + 1], accept_tags=accept_tags,
                         accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                         exclude_tags_start_with=exclude_tags_start_with)
    return v


def overlap_matrix_sentiment(w2v_model, lemma_by_segment, tags_by_segment, accept_tags=[], accept_tags_start_with=[],
                             exclude_tags=[], exclude_tags_start_with=[]):
    v = 0
    hit_elements = 0
    searched_elements = 0
    for index_a, (l_a, t_a) in enumerate(zip(lemma_by_segment, tags_by_segment)):
        v_temp, h_temp, s_temp = sentence_similarity(w2v=w2v_model, sent_a_lemma=l_a, sent_a_tags=t_a,
                                                     sent_b_lemma=lemma_by_segment[index_a + 1],
                                                     sent_b_tags=tags_by_segment[index_a + 1], accept_tags=accept_tags,
                                                     accept_tags_start_with=accept_tags_start_with,
                                                     exclude_tags=exclude_tags,
                                                     exclude_tags_start_with=exclude_tags_start_with)
        v += v_temp
        hit_elements += h_temp
        searched_elements += s_temp
    hitrate = hit_elements / searched_elements
    return v, hitrate
