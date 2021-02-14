from Helper.Helper_functions import mean_of_list, search_tag_set
from Helper.w2v_model import sentence_similarity

def semantic_shift(w2v_model, lemma_by_segment, tags_by_segment, accept_tags=[], accept_tags_start_with=[],
                             exclude_tags=[], exclude_tags_start_with=[]):
    '''
    Ref: Crossley2019 - Semantic similarity features
    :param w2v_model:
    :param lemma_by_segment:
    :param tags_by_segment:
    :param accept_tags:
    :param accept_tags_start_with:
    :param exclude_tags:
    :param exclude_tags_start_with:
    :return:
    '''
    if w2v_model is None:
        return 0, 0
    
    v =[]
    hit_ratio = []
    for index_a, (l_a, t_a) in enumerate(zip(lemma_by_segment, tags_by_segment)):
        if index_a + 1 >= len(lemma_by_segment):
            continue
        v_temp, hr_temp = sentence_similarity(w2v=w2v_model, sent_a_lemma=l_a, sent_a_tags=t_a,
                                                     sent_b_lemma=lemma_by_segment[index_a + 1],
                                                     sent_b_tags=tags_by_segment[index_a + 1], accept_tags=accept_tags,
                                                     accept_tags_start_with=accept_tags_start_with,
                                                     exclude_tags=exclude_tags,
                                                     exclude_tags_start_with=exclude_tags_start_with)
        if v_temp == 0 and hr_temp == 0:
            continue
        v.append(v_temp)
        hit_ratio.append(hr_temp)
    return mean_of_list(v), mean_of_list(hit_ratio)


def tag_overlap_by_sent(sent_a_tags, sent_a_lemma, sent_b_tags, sent_b_lemma, accept_tags=[], accept_tags_start_with=["N"],
                exclude_tags=[], exclude_tags_start_with=[]):
    '''
    Ref: Crossley 2016 - Lexical Overlap
    Ref: Pitler08 - Elements of Lexical cohesion - Generally Bad Features! - has cosine similarity - use conditional probability instead
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

    overlapping_lemma = lemma_set_sent_a.intersection(lemma_set_sent_b)
    
    return len(overlapping_lemma)/len(lemma_set_sent_b)



def tag_overlap(lemma_by_segment, tags_by_segment, accept_tags=[], accept_tags_start_with=[],
                   exclude_tags=[], exclude_tags_start_with=[]):
    '''
    Ref: Crossley 2016 - Lexical Overlap
    Ref: Pitler08 - Elements of Lexical cohesion - Generally Bad Features! - has cosine similarity - use conditional probability instead
    :param lemma_by_segment:
    :param tags_by_segment:
    :param accept_tags:
    :param accept_tags_start_with:
    :param exclude_tags:
    :param exclude_tags_start_with:
    :return:
    '''
    v = []
    for index_a, (l_a, t_a) in enumerate(zip(lemma_by_segment, tags_by_segment)):
        if index_a + 1 >= len(lemma_by_segment):
            continue
        v.append(tag_overlap_by_sent(sent_a_lemma=l_a, sent_a_tags=t_a, sent_b_lemma=lemma_by_segment[index_a + 1],
                         sent_b_tags=tags_by_segment[index_a + 1], accept_tags=accept_tags,
                         accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                         exclude_tags_start_with=exclude_tags_start_with))
    return mean_of_list(v)

def affinity_shift(lemma_by_sent, affinity_dict, affinity_label):
    '''
    Calculate for the given Affinity scores, their absolute sentiment shift, for adjacent sentences
    :param lemma_by_sent: list, a list of sentences, which are a list of lemma
    :param affinity_dict: dictionary with words as identifier and a dict of affinity values as its value
    :param affinity_label: list, of affinity value names
    :return: list, float mean absolute pairwise affinity scores, for given affinity labels
    '''
    affinities = [[]] * len(affinity_label)
    for lemma in lemma_by_sent:
        temp_affinities = [[]] * len(affinity_label)
        for l in lemma:
            temp = affinity_dict.get(l, None)
            if temp is not None:
                for aff_index, aff_lab in enumerate(affinity_label):
                    temp_aff_score = temp_affinities[aff_index]
                    temp_aff_score.append(temp.get(aff_lab, None))
                    temp_affinities[aff_index] = temp_aff_score
        for aff_index, aff_lab in enumerate(affinity_label):
            affinities[aff_index] = mean_of_list(temp_affinities[aff_index])
    affinity_shift_score = {}
    for aff_values, aff_lab in zip(affinities, affinity_label):
        aff_shift = []
        for sent_index, aff_val in enumerate(aff_values):
            if sent_index + 1 >= len(aff_values):
                continue
            aff_shift.append(abs(aff_val - aff_val[sent_index + 1]))
        affinity_shift_score[aff_lab] = mean_of_list(aff_shift)
    affinity_shift_score = {"Affinity shift " + str(k): v for k, v in affinity_shift_score.items()}
    return affinity_shift_score



