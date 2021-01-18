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


# TODO: Dividing by 1000 or something relative
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

# # TODO:
# def count_connective_words(document_words, connective_words=["daher", "darum", "deshalb", "dementsprechend"]):
#   # TODO: Need a larger set of connective words! and maybe devide by those categories
#   #  LEMMA? -> use tags?
#   #  isnt it the conjungtions?
#   #  Connectives:
#   #  (1) clarifying connectives, such as in other words and that is;
#   #  (2) additive connectives, such as also and moreover;
#   #  (3) temporal connectives, such as after, before, and when; and
#   #  (4) causal connectives, such as because, so, and consequently.
#   #  On another dimension, there is a contrast between positive
#   #  and negative connectives. For example, adversative additive
#   #  connectives (e.g., however, in contrast) and adversative
#   #  causal connectives (e.g., although) are negative.
#
#   counter = 0
#   for i in document_words:
#     if i in connective_words:
#       counter += 1
#   return counter
