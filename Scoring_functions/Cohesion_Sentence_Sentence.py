from Helper.Helper_functions import mean_of_list, search_tag_set
from Helper.w2v_model import sentence_similarity

def sentiment_shift(w2v_model, lemma_by_segment, tags_by_segment, accept_tags=[], accept_tags_start_with=[],
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
    return (mean_of_list(v), mean_of_list(hit_ratio))


def tag_overlap(tagset_by_sent, tagset_name):
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

    tagset = tagset_by_sent.get(tagset_name, {})
    v = []
    for index_a, tagset_sent in enumerate(tagset):
        if index_a + 1 >= len(tagset):
            continue
        lemma_set_sent_a = set(tagset_sent.keys())
        lemma_set_sent_b = set(tagset[index_a + 1].keys())

        overlapping_lemma = lemma_set_sent_a.intersection(lemma_set_sent_b)
        if len(lemma_set_sent_b) > 0:
            v.append(len(overlapping_lemma) / len(lemma_set_sent_b))
    return mean_of_list(v)


def affinity_shift(affinity_score_dict, affinity_label):
    
    '''
    Ref: Jacobs2018
    Calculate with the Affinity scores, by sentence, their absolute affinity shift, for adjacent sentences, per affinity_label element
    :param affinity_score_dict: dict{affinity label: list[sentence list[affinity values per lemma]]}
    :param affinity_label: list, of affinity value names
    :return: dict, {key=affinity_labels: value=float, mean absolute difference of pairwise sentence affinity}
    '''
    aff_shift_score = {}
    
    # Iterate over all affinity_labels
    for aff_lab in affinity_label:
        aff_shift = []
        aff_by_sent = affinity_score_dict.get(aff_lab, [])
        
        # calculate pairwise affinity distance
        for sent_index, aff_list in enumerate(aff_by_sent):
            if sent_index + 1 >= len(aff_by_sent):
                continue
            aff_shift.append(abs(sum(aff_list) - sum(aff_by_sent[sent_index + 1])))
        aff_shift_score[aff_lab] = mean_of_list(aff_shift)
        
    # Label affinity shift scores for final table
    aff_shift_score = {"Affinity shift " + str(k): v for k, v in aff_shift_score.items()}
    return aff_shift_score



