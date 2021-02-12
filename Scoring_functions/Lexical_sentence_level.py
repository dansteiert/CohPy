from Helper.Helper_functions import search_tag_set, to_count_dict
import numpy as np

def type_token_ratio(document_tags, accept_tags=[], accept_tags_start_with=["N"], exclude_tags=[],
                     exclude_tags_start_with=[]):
    '''
    Calculate the ratio # types/ sum of tokens. A type is a unique POS, a token is the times a type occures
    :param document_tags:  list, of POS-tags
    :param accept_tags: list, a set of POS-tag, needed for check_tags function
    :param accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: float, ratio # types/ sum of tokens
    '''
    # count unique words against their repetitions.
    # split into Nouns and non-Noun content words
    tag_list_nouns = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                                    accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                                    exclude_tags_start_with=exclude_tags_start_with)
    count_dict = to_count_dict(aggregate_list=tag_list_nouns)
    if len(count_dict) == 0:
        return 0
    else:
        return len(count_dict) / sum(count_dict.values())


# Todo: There are two kinds of lexical diversity implement both!!
def lexical_diversity(document_tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[],
                      exclude_tags_start_with=["$"]):
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


# TODO: check for correctness!
def pronoun_resolution(document_tags, nouns_accept_tags=[], nouns_accept_tags_start_with=["N"], nouns_exclude_tags=[],
                       nouns_exclude_tags_start_with=[],
                       pronouns_accept_tags=["ADJA", "ADJD", "ADV"], pronouns_accept_tags_start_with=["P"],
                       pronouns_exclude_tags=["PTK"],
                       pronouns_exclude_tags_start_with=[]):
    '''
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


# TODO: get Tagsets for Functional and Content words!
def content_functional_ratio(document_tags, content_tags=[], content_tags_start_with=["N"], exclude_content_tags=[],
                             exclude_content_tags_start_with=[], functional_tags=["READUP!"],
                             functional_tags_start_with=[], exclude_functional_tags=[],
                             exclude_functional_tags_start_with=[]):
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


# TODO: Dividing by 1000 or something relative -> Look it up and include, if necessary
def POS_frequency(document_tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[],
                  exclude_tags_start_with=["$"]):
    # where is it used/implemented? - what type should be returned?
    # How fine should the differentiation be done?
    tag_list = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                              accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                              exclude_tags_start_with=exclude_tags_start_with)
    count_dict = to_count_dict(tag_list)
    tag_dict = {key: val / 1000 for (key, val) in count_dict.items()}
    return tag_dict
