from Helper.Helper_functions import search_tag_set, to_count_dict, mean_of_list
import numpy as np




def logical_incidence(aggregate, document_tags, accept_tags=[], accept_tags_start_with=["$"], exclude_tags=[],
                      exclude_tags_start_with=[]):
    '''
    Ref: Grasser2004 - Logical Operators

    :param aggregate: list, list of elements associated to the POS-tags given
    :param tags: list, list of POS-tags, associated to the aggregate list
    :param accept_tags: list, a set of POS-tag, needed for check_tags function
    :param accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return:
    '''
    tag_list = search_tag_set(aggregate=aggregate, tags=document_tags, accept_tags=accept_tags,
                              accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                              exclude_tags_start_with=exclude_tags_start_with)
    count_dict = to_count_dict(tag_list)
    normalizer = len(aggregate) / 1000
    incidence_scores = [v for k, v in count_dict.items()]
    incidence_scores.append(sum(incidence_scores))
    incidence_scores = [i / normalizer for i in incidence_scores]
    return mean_of_list(incidence_scores)


def connective_incidence(lemma, connective_dict, name_positive_connective, name_negative_connective):
    '''
    Ref: Grasser2004 - Connectives
    Ref: Crossley2016- Connectives
    Calculate connective incidence scores
    :param lemma: list, of lemma
    :param connective_dict: dict, with words and their connective category
    :param name_positive_connective: str, name of the positive connective group
    :param name_negative_connective: str, name of the negative connective group
    :return: set(float, float, float), mean incidence of -, incidence of negative-, incidence of positive connectives
    '''
    agg_list = []
    for index, l in enumerate(lemma):
        temp = []
        for i in range(0, 3):
            temp.append(connective_dict.get(" ".join(lemma[index: index + i]), None))
        for i in reversed(temp):
            if i is not None:
                agg_list.append(i)
                break
    count_dict = to_count_dict(agg_list)
    normalizer = len(lemma) / 1000
    connective_count = [v / normalizer for k, v in count_dict.items()]
    connective_names = [k for k, v in count_dict.items()]
    neg_connective_count = [v for k, v in zip(connective_names, connective_count) if k == name_negative_connective]
    pos_connective_count = [v for k, v in zip(connective_names, connective_count) if k == name_positive_connective]
    
    return (mean_of_list(connective_count), neg_connective_count, pos_connective_count)



def Flescher_Reading_Ease(document_words, document_syllables, num_sentences):
    '''
    Ref: Grasser2004 - Readability Scores
    :param document_words:
    :param document_syllables:
    :param num_sentences:
    :return:
    '''
    if len(document_words) < 200:
        return None

    ASL = len(document_words) / num_sentences  # ratio #words/#sent
    ASW = sum(document_syllables) / len(document_words)  # # ratio Syllables/Words
    return 206.835 - 1.015 * ASL - 84.6 * ASW


def Flescher_Kincaid_Grade_Level(document_words, document_syllables, num_sentences):
    '''
    Ref: Grasser2004 - Readability Scores

    :param document_words:
    :param document_syllables:
    :param num_sentences:
    :return:
    '''
    if len(document_words) < 200:
        return None

    ASL = len(document_words) / num_sentences  # ratio #words/#sent
    ASW = sum(document_syllables) / len(document_words)
    return 0.39 * ASL + 11.8 * ASW - 15.59


# def flesch_kincaid_ease(text):
#     """ http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_Reading_Ease
#     Score	School level	Notes
#     100.00-90.00 5th grade	Very easy to read. Easily understood by an average 11-year-old student.
#     90.0–80.0	6th grade	Easy to read. Conversational English for consumers.
#     80.0–70.0	7th grade	Fairly easy to read.
#     70.0–60.0	8th & 9th grade	Plain English. Easily understood by 13- to 15-year-old students.
#     60.0–50.0	10th to 12th grade	Fairly difficult to read.
#     50.0–30.0	College	Difficult to read.
#     30.0–0.0 	College graduate Very difficult to read. Best understood by university graduates.
#     """
#     text = preprocess(text)
#     return 206.835 - (1.015 * avg_words_per_sentence(text)) - (84.6 * avg_syllables_per_word(text))
#
# def FKDE(text):
#     """ http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_Reading_Ease """
#     text = preprocess(text)
#     return 180 - avg_words_per_sentence(text) - (58 * avg_syllables_per_word(text))
#
# def douma(text):
#     """ Variant of Flesch-Kincaid for Dutch: http://www.cnts.ua.ac.be/papers/2002/Geudens02.pdf """
#     text = preprocess(text)
#     return 206.84 - (0.33 * avg_words_per_sentence(text)) - (0.77 * avg_syllables_per_word(text))
#
# def kandel_moles(text):
#     """ Variant of Flesch-Kincaid for French (citation not easily traceable) """
#     text = preprocess(text)
#     return 209 - (1.15 * avg_words_per_sentence(text)) - (0.68 * avg_syllables_per_word(text))
#
# def gulpease(text):
#     """ https://it.wikipedia.org/wiki/Indice_Gulpease """
#     text = preprocess(text)
#     return 89.0 + (300.0 * sentence_count(text) - 10.0 * letter_count(text))/(word_count(text))
#
# def fernandez_huerta(text):
#     """ Developed for Spanish texts (citation not easily traceable) """
#     text = preprocess(text)
#     factor = 100.0 / word_count(text)
#     return 206.84 - (0.6 * factor * syllable_count(text)) - (1.02 * factor * sentence_count(text))
#
#
# """
# Grade level estimators: higher scores imply more advanced-level material
# """
#
# def flesch_kincaid_grade(text):
#     """ http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch.E2.80.93Kincaid_Grade_Level """
#     text = preprocess(text)
#     return (0.39 * avg_words_per_sentence(text)) + (11.8 * avg_syllables_per_word(text)) - 15.59
#
# def gunning_fog(text):
#     """ http://en.wikipedia.org/wiki/Gunning_Fog_Index """
#     text = preprocess(text)
#     return 0.4 * (avg_words_per_sentence(text) + percent_three_syllable_words(text, False))
#
# def coleman_liau(text):
#     """ http://en.wikipedia.org/wiki/Coleman-Liau_Index """
#     text = preprocess(text)
#     return  (5.89 * letter_count(text) / word_count(text)) - (0.3 * sentence_count(text) / word_count(text)) - 15.8
#
# def smog(text):
#     """ http://en.wikipedia.org/wiki/SMOG_Index """
#     text = preprocess(text)
#     return 1.043 * sqrt((three_syllable_word_count(text) * (30.0 / sentence_count(text))) + 3.1291)
#
# def ari(text):
#     """ http://en.wikipedia.org/wiki/Automated_Readability_Index """
#     text = preprocess(text)
#     return (4.71 * letter_count(text) / word_count(text)) + (0.5 * word_count(text) / sentence_count(text)) - 21.43
#
# """
# Other indices (not grade levels): higher scores imply "more difficult" reading
# """
#
# def lix(text):
#     """ http://en.wikipedia.org/wiki/LIX """
#     text = preprocess(text)
#     num_words = word_count(text)
#     return (100.0 * six_letter_word_count(text) / num_words) + (1.0 * num_words / sentence_count(text))
#
# def rix(text):
#     """ More generalized variant of LIX """
#     text = preprocess(text)
#     return 1.0 * six_letter_word_count(text) / sentence_count(text)

