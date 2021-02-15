from Helper.Helper_functions import search_tag_set, to_count_dict, mean_of_list
import numpy as np


def type_token_ratio(tagsets_by_doc, tagset_name):
    '''
    Ref: Grasser2004 - Type:Token Ratio
    Ref: Crossley2016 - Type-token ratio
    Ref Jacobs2018 - Study 2 - adds Logarithm
    Calculate the ratio # types/ sum of tokens. A type is a unique word, a token is the times a type occures
    :param document_lemma: list, of lemma
    :param document_tags:  list, of POS-tags
    :param accept_tags: list, a set of POS-tag, needed for check_tags function
    :param accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: float, ratio # types/ sum of tokens
    '''

    tagset_dict = tagsets_by_doc.get(tagset_name, {})
        
    type = len(tagset_dict)
    token = sum(tagset_dict.values())
    if token > 0:
        return type/token
    else:
        return np.Infinity


# Todo: There are two kinds of lexical diversity implement both!!
def lexical_diversity(tagsets_by_sentence, tagset_name):
    '''
    Similar to Type-Token Ratio, it searches for types, unique occuring POS elements
    :param document_tags: list, a set of POS-tag, needed for check_tags function
    :param accept_tags: list, a set of POS-tag, needed for check_tags function
    :param accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: int, Number of Types
    '''
    
    
    tag_list = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                              accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                              exclude_tags_start_with=exclude_tags_start_with)
    count_dict = to_count_dict(aggregate_list=tag_list)
    lexical_terms = [k for k, v in count_dict.items() if v > 0]
    return len(lexical_terms)


def noun_pronoun_proportion(document_tags, nouns_accept_tags=[], nouns_accept_tags_start_with=["N"], nouns_exclude_tags=[],
                       nouns_exclude_tags_start_with=[],
                       pronouns_accept_tags=["ADJA", "ADJD", "ADV"], pronouns_accept_tags_start_with=["P"],
                       pronouns_exclude_tags=["PTK"],
                       pronouns_exclude_tags_start_with=[]):
    '''
    Ref: Grasser 2004 - Density Scores
    Ref: Crossley 2016 - Givenness
    Calculate the ratio of Nouns to Pronouns
    :param document_tags: list, a set of POS-tag, needed for check_tags function
    :param nouns_accept_tags: list, a set of POS-tag, needed for check_tags function
    :param nouns_accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param nouns_exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param nouns_exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param pronouns_accept_tags: list, a set of POS-tag, needed for check_tags function
    :param pronouns_accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param pronouns_exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param pronouns_exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: float, the ratio of Nouns/Pronouns
    '''

    tag_list_nouns = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=nouns_accept_tags,
                                    accept_tags_start_with=nouns_accept_tags_start_with,
                                    exclude_tags=nouns_exclude_tags,
                                    exclude_tags_start_with=nouns_exclude_tags_start_with)
    tag_list_pronouns = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=pronouns_accept_tags,
                                       accept_tags_start_with=pronouns_accept_tags_start_with,
                                       exclude_tags=pronouns_exclude_tags,
                                       exclude_tags_start_with=pronouns_exclude_tags_start_with)
    if len(tag_list_pronouns) > 0:
        return len(tag_list_nouns) / len(tag_list_pronouns)
    return np.Infinity


def content_functional_ratio(document_tags, content_tags=[], content_tags_start_with=[], exclude_content_tags=[],
                             exclude_content_tags_start_with=[], functional_tags=[],
                             functional_tags_start_with=[], exclude_functional_tags=[],
                             exclude_functional_tags_start_with=[]):
    '''
    Ref: CohMetrix. Grasser 2004 - Part of Speech
    A ratio of Content to functional POS elements is calculated.
    :param document_tags: list, of tags
    :param content_tags: list, a set of POS-tag, needed for check_tags function
    :param content_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_content_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_content_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param functional_tags: list, a set of POS-tag, needed for check_tags function
    :param functional_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_functional_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_functional_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: float, ratio of content to functional POS elements
    '''
    tag_list_content = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=content_tags,
                                      accept_tags_start_with=content_tags_start_with, exclude_tags=exclude_content_tags,
                                      exclude_tags_start_with=exclude_content_tags_start_with)
    tag_list_functional = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=functional_tags,
                                         accept_tags_start_with=functional_tags_start_with,
                                         exclude_tags=exclude_functional_tags,
                                         exclude_tags_start_with=exclude_functional_tags_start_with)

    if len(tag_list_functional) > 0:
        return len(tag_list_content) / len(tag_list_functional)
    return len(tag_list_content)


def POS_frequency(document_tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[],
                  exclude_tags_start_with=["$"]):
    '''
    Ref: Graesser2004??
    :param document_tags:
    :param accept_tags:
    :param accept_tags_start_with:
    :param exclude_tags:
    :param exclude_tags_start_with:
    :return:
    '''
    tag_list = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                              accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                              exclude_tags_start_with=exclude_tags_start_with)
    count_dict = to_count_dict(tag_list)
    normalizer = len(document_tags)/1000
    tag_dict = {key: val / normalizer for (key, val) in count_dict.items()}
    return tag_dict


def ratio_tags_a_to_tags_b(tagsets_by_doc, tagset_a, tagset_b):
    '''
    Ref: CohMetrix. Grasser 2004 - Part of Speech
    A ratio of Content to functional POS elements is calculated.
    :param document_tags: list, of tags
    :param content_tags: list, a set of POS-tag, needed for check_tags function
    :param content_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_content_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_content_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param functional_tags: list, a set of POS-tag, needed for check_tags function
    :param functional_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_functional_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_functional_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: float, ratio of content to functional POS elements
    '''
    
    tags_a = tagsets_by_doc.get(tagset_a, {})
    tags_b = tagsets_by_doc.get(tagset_b, {})
    if len(tags_b) > 0:
        return sum(tags_a.values()) / sum(tags_b.values())
    else:
        return np.Infinity