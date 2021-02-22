from Helper.Helper_functions import search_tag_set, to_count_dict, mean_of_list
import numpy as np


def type_token_ratio(tagsets_by_doc, tagset_name):
    '''
    Ref: Grasser2004 - Type:Token Ratio
    Ref: Crossley2016 - Type-token ratio
    Ref Jacobs2018 - Study 2 - adds Logarithm
    Calculate the ratio # types/ sum of tokens. A type is a unique word, a token is the times a type occures
    :param tagsets_by_doc: dict, {key=tagset names, value=dict{key=lemma, value=absolute word count}}
    :param tagset_name: str, tagset name for which this ratio should be calculated
    :return: float, type/tokens
    '''
    tagset_dict = tagsets_by_doc.get(tagset_name, {})
        
    type = len(tagset_dict)
    token = sum(tagset_dict.values())
    if token > 0:
        return type/token
    else:
        return np.Infinity


def lexical_diversity(word_frequency_dict, document_sentences):
    '''
    Similar to Type-Token Ratio, it searches for types, unique occuring POS elements, similar to type token ratio, but with the number of sentences as divisor
    Halliday Lexical density.
    :param word_frequency_dict: dict, {key=word: value=absolute freqeuncy
    :param document_sentences: int, # of sentences in the document
    :return: float, ratio of # lexical items to # clauses -> here for simplicity, sentences
    '''
    lex_diversity = len(word_frequency_dict)/document_sentences
    return lex_diversity


def ratio_tags_a_to_tags_b(tagsets_by_doc, tagset_a, tagset_b):
    '''
    Ref: CohMetrix. Grasser 2004 - Part of Speech
    Ref: Jacobs 2018 - Adjective-Verb Quotient
    Ref: Grasser 2004 - Density Scores
    Ref: Crossley 2016 - Givenness
    A ratio of elements in tagset a to elements in tagset b is calculated. E.g. content-functional-ratio, Adjective-Verb Quotient, Pronoun-Noun ratio
    :param tagsets_by_doc: dict, {key=tagset names, value=dict{key=lemma, value=absolute word count}}
    :param tagset_a: str, tagset name for which this ratio should be calculated
    :param tagset_b: str, tagset name for which this ratio should be calculated
    :return: float, # elements in a/ # elements in b
    '''
    
    tags_a = tagsets_by_doc.get(tagset_a, {})
    tags_b = tagsets_by_doc.get(tagset_b, {})
    if len(tags_b) > 0:
        return sum(tags_a.values()) / sum(tags_b.values())
    else:
        return np.Infinity
    