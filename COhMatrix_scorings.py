import pandas as pd
import os
import numpy as np

def word_length(document_word):
  '''

  :param document: List of tokenized entries
  :return: length for each entry in the list
  '''
  return [len(i) for i in document_word]


def word_familarity(document_word, familarity_dict):
  # retrieve frequency from a DB of Tests
  count = 0
  for i in document_word:
    temp = familarity_dict.get(i, None)
    if temp is not None:
      count += temp
  count /= len(document_word)
  return count


def syllable_count(document_word):
  '''

  :param document: List of tokenized entries
  :return: Syllable count for each entry in the list
  '''
  syllable_list = []
  for i in document_word:
    count = 0
    flipper = False
    for j in i.lower():
      if flipper:
        flipper= False
        continue
      if j in "aeiou":
        count += 1
        flipper = True
    syllable_list.append(count)
  return syllable_list


# TODO: pronoun resolution
def pronoun_resolution(document_tags, noun_tags=[], noun_tags_start_with=["N"], exclude_noun_tags=[], 
                       exclude_noun_tags_start_with =[], 
                       pronoun_tags=["ADJA", "ADJD", "ADV"], pronoun_tags_start_with=["P"], exclude_pronoun_tags=["PTK"],
                       exclude_pronoun_tags_start_with =[]):
                       
  # This Task, wants to map pronouns with their respectiv "owners"
  # more complex task!
  # Pronoun density  consists  of  the  proportion  of  noun  phrases(NPs, as defined  by  a  syntactic
  # parser, which  will  be  described  later) that  are  captured by  pronouns(as defined  by  the  Brill  POS  tagger).

  tag_list_nouns = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=noun_tags,
                                  accept_tags_start_with=noun_tags_start_with, exclude_tags=exclude_noun_tags,
                                  exclude_tags_start_with=exclude_noun_tags_start_with)
  tag_list_pronouns = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=pronoun_tags,
                                  accept_tags_start_with=pronoun_tags_start_with, exclude_tags=exclude_pronoun_tags,
                                  exclude_tags_start_with=exclude_pronoun_tags_start_with)
  if len(tag_list_pronouns) > 0:
    return len(tag_list_nouns)/len(tag_list_pronouns)
  return len(tag_list_nouns)


# TODO: causal cohesion
def casual_cohesion(document):
  # The total list of causal particles comes either from this short list of verbs or from the causal conjunctions,
  # transitional adverbs, and causal connectives. The current metric of causal cohesion, which is a primary measure,
  # is simply a ratio of causal particles (P) to causal verbs (V).

  return None


def count_logicals(document_tags, accept_tags=["KON", "KOKOM"], accept_tags_start_with=[],
                          exclude_tags=[], exclude_tags_start_with=[]):
  # high count means more "work"
  # KOUI:subordinating  conjunction  followed|  by \zu" and innitive um [zu leben], anstatt [zu fragen]
  # KOUS:  subordinating  conjunction  followed  by  clause  weil, dass, damit, wenn, ob
  # KON:  coordinating  conjunction  und, oder, aber
  # KOKOM:  comparative  conjunction  als, wie
  logical_count = sum(search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                                    accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                                    exclude_tags_start_with=exclude_tags_start_with))  
  return logical_count


def mean_sentence_length(document_tags, accept_tags=["$."], accept_tags_start_with=[],
                         exclude_tags=[], exclude_tags_start_with=[]):
    sent_length = []
    temp_sent_length = 0
    for t in document_tags:
        if check_tags(tag=t, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                      exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with):
            sent_length.append(temp_sent_length)
        else:
            temp_sent_length += 1
    return mean_of_list(sent_length)



def Flescher_Reading_Ease(document_words, document_tags, document_syllables, accept_tags=["$."], accept_tags_start_with=[], 
                          exclude_tags=[], exclude_tags_start_with=[]):
  if len(document_words) < 200:
    return None
  num_sentences = sum(search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                                    accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                                    exclude_tags_start_with=exclude_tags_start_with))  
  ASL = len(document_words)/num_sentences # ratio #words/#sent
  ASW = sum(document_syllables)/len(document_words) # # ratio Syllables/Words
  return 206.835 - 1.015 * ASL - 84.6 * ASW


def Flescher_Kincaid_Grade_Level(document_words, document_tags, document_syllables, accept_tags=["$."], 
                                 accept_tags_start_with=[], exclude_tags=[], exclude_tags_start_with=[]):
  if len(document_words) < 200:
    return None
  num_sentences = sum(search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                                    accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                                    exclude_tags_start_with=exclude_tags_start_with))
  
  ASL = len(document_words)/num_sentences # ratio #words/#sent
  ASW = sum(document_syllables)/len(document_words)
  return 0.39*ASL + 11.8*ASW - 15.59





def POS_frequency(document_tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[],
                      exclude_tags_start_with=["$"]):
  # where is it used/implemented? - what type should be returned?
  # How fine should the differentiation be done?
  tag_list= search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                                    accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                                    exclude_tags_start_with=exclude_tags_start_with)
  count_dict = to_count_dict(tag_list)
  tag_dict = {key: val/1000 for (key, val) in count_dict.items()}
  return tag_dict


def connective_words(document_words,
                     connective_words=["daher", "darum", "deshalb", "dementsprechend"]):
  # TODO: Need a larger set of connective words! and maybe devide by those categories
  #  LEMMA? -> use tags?
  #  Connectives:
  #  (1) clarifying connectives, such as in other words and that is;
  #  (2) additive connectives, such as also and moreover;
  #  (3) temporal connectives, such as after, before, and when; and
  #  (4) causal connectives, such as because, so, and consequently.
  #  On another dimension, there is a contrast between positive
  #  and negative connectives. For example, adversative additive
  #  connectives (e.g., however, in contrast) and adversative
  #  causal connectives (e.g., although) are negative.

  counter = 0
  for i in document_words:
    if i in connective_words:
      counter += 1
  return counter




# TODO:
### Things to include:

#   COmbine content words (Nouns, lexical verbs, adjectives and adverbs) and functional words (the rest?), and calculate this indice again
# NP, VP: Verb oder Substantive des Satzes  ist Kopf des Satzes (wichtigester bestandteil) Willkür?!
def content_functional_ratio(document_tags, content_tags=[], content_tags_start_with=["N"], exclude_content_tags=[],
                             exclude_content_tags_start_with =[], functional_tags=["READUP!"],
                             functional_tags_start_with=[], exclude_functional_tags=[],
                             exclude_functional_tags_start_with =[]):  
  tag_list_content= search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=content_tags,
                                    accept_tags_start_with=content_tags_start_with, exclude_tags=exclude_content_tags,
                                    exclude_tags_start_with=exclude_content_tags_start_with)
  tag_list_functional= search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=functional_tags,
                                    accept_tags_start_with=functional_tags_start_with, exclude_tags=exclude_functional_tags,
                                    exclude_tags_start_with=exclude_functional_tags_start_with)

  if len(tag_list_functional) > 0:
    return len(tag_list_content)/len(tag_list_functional)
  return len(tag_list_content)


def type_token_ratio(document_tags, noun_tags=[], noun_tags_start_with=["N"], exclude_noun_tags=[],
                             exclude_noun_tags_start_with =[], non_noun_tags=["ADJA", "ADJD", "ADV"],
                             non_noun_tags_start_with=["V"], exclude_non_noun_tags=[],
                             exclude_non_noun_tags_start_with =[]):
  # count unique words against their repetitions.
  # split into Nouns and non-Noun content words
  tag_list_nouns= search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=noun_tags,
                                    accept_tags_start_with=noun_tags_start_with, exclude_tags=exclude_noun_tags,
                                    exclude_tags_start_with=exclude_noun_tags_start_with)
  tag_list_non_nouns= search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=non_noun_tags,
                                    accept_tags_start_with=non_noun_tags_start_with, exclude_tags=exclude_non_noun_tags,
                                    exclude_tags_start_with=exclude_non_noun_tags_start_with)
  count_dict = to_count_dict(aggregate_list=tag_list_nouns)
  count_dict_non_nouns = to_count_dict(aggregate_list=tag_list_non_nouns)


  if len(count_dict) == 0:
    ratio = 0
  else:
    ratio = len(count_dict)/sum(count_dict.values())
  if len(count_dict_non_nouns) == 0:
    ratio_none = 0
  else:
    ratio_none = len(count_dict_non_nouns)/sum(count_dict_non_nouns.values())
  return (ratio, ratio_none)


def count_puncutation(document_tags, accept_tags=[], accept_tags_start_with=["$"], exclude_tags=[],
                      exclude_tags_start_with=[]):
  tag_list= search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                                    accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                                    exclude_tags_start_with=exclude_tags_start_with)
  return len(tag_list)/len(document_tags)


def tag_overlap(sent_a_tags, sent_a_lemma, sent_b_tags, sent_b_lemma, accept_tags=["NE", "NN"], accept_tags_start_with=["N"],
                exclude_tags=[], exclude_tags_start_with=[]):
  '''
  :param sent_a_tags: tag list of sentance/ pargraph/.. a
  :param sent_a_lemma: lemma list of sentance/ pargraph/.. a
  :param sent_b_tags: tag list of sentance/ pargraph/.. b
  :param sent_b_lemma: lemma list of sentance/ pargraph/.. b
  :param tags: choose a set of tags, for which the overlap should be calculated
  :return: number of overlapping lemma
  '''
  lemma_set_sent_a = set(search_tag_set(aggregate=sent_a_lemma, tags=sent_a_tags, accept_tags=accept_tags,
                                        accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                                        exclude_tags_start_with=exclude_tags_start_with))
  lemma_set_sent_b = set(search_tag_set(aggregate=sent_b_lemma, tags=sent_b_tags, accept_tags=accept_tags,
                                        accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                                        exclude_tags_start_with=exclude_tags_start_with))

  overlapping_lemma = lemma_set_sent_a.intersect(lemma_set_sent_b)
  return len(overlapping_lemma)


def word_repetition(document_lemma, document_tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=["ART"], exclude_tags_start_with = ["$"]):
  # is it important how often a word occured multiple times
  lemma_list = search_tag_set(aggregate=document_lemma, tags=document_tags, accept_tags=accept_tags,
                            accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                            exclude_tags_start_with=exclude_tags_start_with)
  count_dict = to_count_dict(aggregate_list=lemma_list)

  repeated_words = [k for k, v in count_dict.items() if v > 1]
  return len(repeated_words)


def lexical_diversity(document_tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[], exclude_tags_start_with=["$"]):
  tag_list = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags, 
                            accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags, 
                            exclude_tags_start_with=exclude_tags_start_with)
  count_dict = to_count_dict(aggregate_list=tag_list)
  lexical_terms = [k for k, v in count_dict.items() if v > 0]
  return len(lexical_terms)

################# Helper functions
def check_tags(tag, accept_tags=[], accept_tags_start_with=[], exclude_tags=[], exclude_tags_start_with=[]):
  if tag in accept_tags or (tag[0] in accept_tags_start_with and subtag_matching(tag=tag, subtags=exclude_tags)):
        return True
  if tag not in exclude_tags or (tag[0] not in exclude_tags_start_with or (subtag_matching(tag=tag, subtags=accept_tags))):
      return True
  return False


def subtag_matching(tag, subtags):
  for st in subtags:
    if st in tag:
      return True
  return False

def search_tag_set(aggregate, tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[], exclude_tags_start_with=[]):
  aggregate_list = [a for a, t in zip(aggregate, tags) 
                    if check_tags(tag=t, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                                  exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with)]
  return aggregate_list

def to_count_dict(aggregate_list):
  count_dict = {}
  for i in aggregate_list:
    count_dict[i] = count_dict.get(i, 0) + 1
  return count_dict




 # TODO: Excluded are Noun Phrases (lack of implementation/knowledge)
def co_reference_matrix(document_tag, document_lemma,  accept_tags=[], accept_tags_start_with=["N", "P"], exclude_tags=["PTK"],
                      exclude_tags_start_with=[], punctuation_tag_group=["$."]):
  # TODO: redo/rethink

  word_dict = {}
  sentence_count = 0
  for l, t in zip(document_lemma, document_tag):
    if check_tags(tag=t, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                  exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with):
      temp_dict = word_dict.get(l, [])
      temp_dict.append(sentence_count)
      word_dict[l] = temp_dict
    elif t in punctuation_tag_group:
      sentence_count += 1
  if sentence_count < 2:
    return None

  # word_list = word_dict.keys()
  co_reference_exists = np.zeros(shape=(sentence_count, sentence_count))
  co_reference_dist = np.zeros(shape=(sentence_count, sentence_count))
  for val in word_dict.values():
    for index, i in enumerate(val):
      try:
        for index_j, j in enumerate(val[index + 1:]):
          co_reference_exists[i, j] = 1
          co_reference_dist[i, j] = 1/abs(i-j)
      except:
        pass
  local_corefererence_cohesion = (1/(sentence_count-1)) * sum([i[index + 1] for index, i in enumerate(co_reference_exists) if index + 1 < len(co_reference_exists)])
  global_corefererence_cohesion = (1/(sentence_count  *((sentence_count-1)/2))) * np.sum(co_reference_exists)
  co_reference_dist_sum = np.sum(co_reference_dist) * 1/sentence_count
  # print(local_corefererence_cohesion, global_corefererence_cohesion, co_reference_dist_sum)
  # print(co_reference_dist)
  # print(co_reference_exists)
  return (local_corefererence_cohesion, global_corefererence_cohesion, co_reference_dist_sum)

def mean_of_list(l):
    return sum(l)/len(l)