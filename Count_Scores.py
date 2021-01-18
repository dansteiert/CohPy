from Helper_functions import *


def count_tags(document_tags, accept_tags=[], accept_tags_start_with=["$"], exclude_tags=[],
               exclude_tags_start_with=[]):
    tag_list = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                              accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                              exclude_tags_start_with=exclude_tags_start_with)
    return len(tag_list)


def syllable_count(document_words):
    '''
  
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


def word_length(document_word):
    '''
  
    :param document: List of tokenized entries
    :return: length for each entry in the list
    '''
    return mean_of_list([len(i) for i in document_word])


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


def lexical_diversity(document_tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[],
                      exclude_tags_start_with=["$"]):
    tag_list = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                              accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                              exclude_tags_start_with=exclude_tags_start_with)
    count_dict = to_count_dict(aggregate_list=tag_list)
    lexical_terms = [k for k, v in count_dict.items() if v > 0]
    return len(lexical_terms)


def word_repetition(document_lemma, document_tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[],
                    exclude_tags_start_with=[]):
    tag_list = search_tag_set(aggregate=document_lemma, tags=document_tags, accept_tags=accept_tags,
                              accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                              exclude_tags_start_with=exclude_tags_start_with)
    count_dict = to_count_dict(aggregate_list=tag_list)
    repeated_terms = [v for k, v in count_dict.items() if v > 1]
    # print(repeated_terms)
    count_repeated_words = len(repeated_terms)
    num_word_repetitions = sum(repeated_terms)
    return (count_repeated_words, num_word_repetitions)

