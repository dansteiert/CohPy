from Helper.Helper_functions import search_tag_set, to_count_dict
import numpy as np


def word_frequency(document_lemma, document_tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[],
                    exclude_tags_start_with=[]):
    '''
    Count unique words and their repetitions and calculate the ratio  Uniquewords by their repetitions.
    The closer to 1 this ratio, the harder to read is the text. Infinity, means that no such value could be calculated.
    :param document_lemma: list, list of Lemma
    :param document_tags: list, list of POS-tags
    :param accept_tags: list, a set of POS-tag, needed for check_tags function
    :param accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: set(int, int, float), Unique word cound, Sum of repetitions, ratio of Uniquewords by their repetitions
    '''
    tag_list = search_tag_set(aggregate=document_lemma, tags=document_tags, accept_tags=accept_tags,
                              accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                              exclude_tags_start_with=exclude_tags_start_with)
    count_dict = to_count_dict(aggregate_list=tag_list)
    repeated_terms = [v for k, v in count_dict.items() if v >= 1]
    count_repeated_words = len(repeated_terms)
    num_word_repetitions = sum(repeated_terms)
    if num_word_repetitions > 0:
        return (count_repeated_words, num_word_repetitions)
    else:
        return np.Infinity


def Flescher_Reading_Ease(document_words, document_syllables, num_sentences):
    if len(document_words) < 200:
        return None

    ASL = len(document_words) / num_sentences  # ratio #words/#sent
    ASW = sum(document_syllables) / len(document_words)  # # ratio Syllables/Words
    return 206.835 - 1.015 * ASL - 84.6 * ASW


def Flescher_Kincaid_Grade_Level(document_words, document_syllables, num_sentences):
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

