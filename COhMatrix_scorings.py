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
def pronoun_resolution(document):
  # This Task, wants to map pronouns with their respectiv "owners"
  # more complex task!
  # Pronoun density  consists  of  the  proportion  of  noun  phrases(NPs, as defined  by  a  syntactic
  # parser, which  will  be  described  later) that  are  captured by  pronouns(as defined  by  the  Brill  POS  tagger).


  # Simpler ratio of Pronouns to Nouns?
  return None



# TODO: causal cohesion
def casual_cohesion(document):
  # The total list of causal particles comes either from this short list of verbs or from the causal conjunctions,
  # transitional adverbs, and causal connectives. The current metric of causal cohesion, which is a primary measure,
  # is simply a ratio of causal particles (P) to causal verbs (V).

  return None


def count_logicals(document_token, logical_tags=["KOUI", "KOUS", "KON", "KOKOM"]):
  # high count means more "work"
  # KOUI:subordinating  conjunction  followed|  by \zu" and innitive um [zu leben], anstatt [zu fragen]
  # KOUS:  subordinating  conjunction  followed  by  clause  weil, dass, damit, wenn, ob
  # KON:  coordinating  conjunction  und, oder, aber
  # KOKOM:  comparative  conjunction  als, wie
  log_count = 0
  for i in document_token:
    if i in logical_tags:
      log_count += 1
  return log_count


def type_token_ratio(document_token, noun_tags=["NN", "NE"], other_tags_to_search=["ADJA", "ADJD", "ADV"], other_tag_groups=["V"]  ):
  # count unique words against their repetitions.
  # split into Nouns and non-Noun content words
  counter_dict = {}
  counter_dict_non_Noun = {}
  for i in document_token:
    if i in noun_tags:
      counter_dict[i] = counter_dict.get(i, 0) + 1
    elif i in other_tags_to_search or i[0] in other_tag_groups: #, "VVFIN, VVIMP", "VVINF", "VVIZU", "VVPP"]
      counter_dict_non_Noun[i] = counter_dict_non_Noun.get(i, 0) + 1

  if len(counter_dict) == 0:
    ratio = 0
  else:
    ratio = len(counter_dict)/sum(counter_dict.values())
  if len(counter_dict_non_Noun) == 0:
    ratio_nonN = 0
  else:
    ratio_nonN = len(counter_dict_non_Noun)/sum(counter_dict_non_Noun.values())


  return (ratio, ratio_nonN)


def Flescher_Reading_Ease(document_words, document_tags, document_syllables, punctuation_tag_group=["$."]):
  if len(document_words) < 200:
    return None
  num_sentences = sum([document_tags.count(i) for i in punctuation_tag_group])
  ASL = len(document_words)/num_sentences # ratio #words/#sent
  ASW = sum(document_syllables)/len(document_words) # # ratio Syllables/Words
  return 206.835 - 1.015 * ASL - 84.6 * ASW

def Flescher_Kincaid_Grade_Level(document_words, document_tags, document_syllables, punctuation_tag_group=["$."]):
  if len(document_words) < 200:
    return None
  num_sentences = sum([document_tags.count(i) for i in punctuation_tag_group])
  ASL = len(document_words)/num_sentences # ratio #words/#sent
  ASW = sum(document_syllables)/len(document_words)
  return 0.39*ASL + 11.8*ASW - 15.59


 # TODO: Excluded are Noun Phrases (lack of implementation/knowledge)
def co_reference_matrix(document_tag, document_lemma, punctuation_tag_group=["$."], tags_to_search=["NN", "NE"], tag_groups_to_search=["P"], tag_subgroups_to_exclude=["T"]):
  word_dict = {}
  sentence_count = 0
  for l, t in zip(document_lemma, document_tag):
    if t in tags_to_search or (t[0] in tag_groups_to_search and not t[1] in tag_subgroups_to_exclude):
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


def POS_frequency(document_tags, tag_exclusions=[], tag_exclusions_groups=["$"]):
  # How fine should the differentiation be done?

  tag_dict = {}
  for i in document_tags:
    if i[0] in tag_exclusions_groups: # exclude punctuations
      continue
    elif i in tag_exclusions:
      continue
    tag_dict[i] = tag_dict.get(i, 0) + 1
  tag_dict = {key: val/1000 for (key, val) in tag_dict.items()}
  return tag_dict


def connective_words(document_words,
                     connective_words=["daher", "darum", "deshalb", "dementsprechend"]):
  # TODO: Need a larger set of connective words! and maybe devide by those categories
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
def content_functional_ratio(document_tags, content_tags=["NN", "NE"], functional_tags=["READUP!"]):
  # similar to token_type_Ratio?
  counter_content = 0
  counter_functional = 0
  for i in document_tags:
    if i in content_tags:
      counter_content += 1
    elif i in functional_tags:
      counter_functional += 1

  # correct ratio
  try:
    return counter_content/counter_functional
  except:
    return counter_content
