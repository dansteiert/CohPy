from Helper.Helper_functions import mean_of_list, search_tag_set
from Helper.w2v_model import sentence_sentiment_shift
import numpy as np

def sentiment_shift(tagset_by_sent, tagset_name, sentiment_dict):
    """
    Ref: Crossley2019 - Semantic similarity features
    :param tagset_by_sent: list[ dict{word: occurance in sentence}], a list of dictionaries, where the keys are the words and the value is the total occurance in this sentence
    :param tagset_name: str, name of the POS-tag set
    :param sentiment_dict: dict, {word: sentiment vector}
    :return: float, mean sentiment shift - cosine distance between sentences
    """

    if sentiment_dict is None:
        return 0
    
    tagset = tagset_by_sent.get(tagset_name, [])
    if len(tagset) == 0:
        print("Empty tagset for ", tagset_name)
    v = []
    for index_a, tagset_sent in enumerate(tagset):
        if index_a + 1 >= len(tagset):
            continue

        v_temp = sentence_sentiment_shift(sent_a_dict=tagset_sent, sent_b_dict=tagset[index_a + 1], sentiment_dict=sentiment_dict)
        if v_temp is None:
            continue
        v.append(v_temp)
    return mean_of_list(v)


def tag_overlap(tagset_by_sent, tagset_name):
    """
    Ref: Crossley 2016 - Lexical Overlap
    Ref: Pitler08 - Elements of Lexical cohesion - Generally Bad Features! - has cosine similarity - use conditional probability instead
    :param tagset_by_sent: list[ dict{word: occurance in sentence}], a list of dictionaries, where the keys are the words and the value is the total occurance in this sentence
    :param tagset_name: str, name of the POS-tag set, for which overlapping lemma should be searched
    :return: float, mean of overlapping elements for given tagset
    """

    # print("Tag OVerlap")
    tagset = tagset_by_sent.get(tagset_name, [])
    if len(tagset) == 0:
        print("Empty tagset for ", tagset_name)
    v = []
    for index_a, tagset_sent in enumerate(tagset):
        if index_a + 1 >= len(tagset):
            continue
        lemma_set_sent_a = set(tagset_sent.keys())
        lemma_set_sent_b = set(tagset[index_a + 1].keys())

        overlapping_lemma = lemma_set_sent_a.intersection(lemma_set_sent_b)
        # print(lemma_set_sent_a, lemma_set_sent_b, overlapping_lemma)
        if len(lemma_set_sent_b) > 0:
            v.append(len(overlapping_lemma) / len(lemma_set_sent_b))
    return mean_of_list(v)


def affective_shift(affective_score_dict, affective_label):
    """
    Ref: Jacobs2018
    Calculate with the affective scores, by sentence, their absolute affective shift, for adjacent sentences, per affective_label element
    :param affective_score_dict: dict{affective label: list[sentence list[affective values per lemma]]}
    :param affective_label: list, of affective value names
    :return: dict, {key=affective_labels: value=float, mean absolute difference of pairwise sentence affective}
    """
    if affective_score_dict is None:
        return None
    aff_shift_score = {}
    
    # Iterate over all affective_labels
    for aff_lab in affective_label:
        aff_shift = []
        aff_by_sent = affective_score_dict.get(aff_lab, [])
        
        # calculate pairwise affective distance
        for sent_index, aff_list in enumerate(aff_by_sent):
            if sent_index + 1 >= len(aff_by_sent):
                continue
            aff_shift.append(abs(sum(aff_list) - sum(aff_by_sent[sent_index + 1])))
        aff_shift_score[aff_lab] = mean_of_list(aff_shift)
        
    # Label affective shift scores for final table
    aff_shift_score = {"affective shift " + str(k): v for k, v in aff_shift_score.items()}
    return aff_shift_score


def tense_change(tagset_by_sent, tagset_name_past="Past", tagset_name_present="Present"):
    """
    Ref:
    Calculate mean number of time changes between adjacent sentences with POS tags
    :param tagset_by_sent: list[ dict{word: occurance in sentence}], a list of dictionaries, where the keys are the words and the value is the total occurance in this sentence
    :param tagset_name_past: str, name of the POS-tag set, which contains verbs in the past form
    :param tagset_name_present: str, name of the POS-tag set, which contains verbs in the present form
    :return: float, mean of times, the tense was changed from past to present or present to past
    """
    
    tagset_past = tagset_by_sent.get(tagset_name_past, [])
    tagset_present = tagset_by_sent.get(tagset_name_present, [])
    v = []
    for index_a, (past_dict, present_dict) in enumerate(zip(tagset_present, tagset_past)):
        if index_a + 1 >= len(tagset_present):
            continue
        
        if len(present_dict) > len(past_dict):
            time_a = 1  # present
        elif len(present_dict) < len(past_dict):
            time_a = -1  # past
        else:
            time_a = 0  # unclear
        
        sent_b_past_dict = tagset_past[index_a + 1]
        sent_b_present_dict = tagset_present[index_a + 1]
        
        if len(sent_b_present_dict) > len(sent_b_past_dict):
            time_b = 1  # present
        elif len(sent_b_present_dict) < len(sent_b_past_dict):
            time_b = -1  # past
        else:
            time_b = 0  # unclear
        
        
        if time_a == 0 and time_b == 0:
            v.append(False)
        else:
            v.append(time_a == time_b)

    return mean_of_list(v)

