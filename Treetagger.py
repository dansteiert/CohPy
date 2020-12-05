import pandas as pd
import treetaggerwrapper as tt
import os
import numpy as np
def CohMatrix():
  t_tagger = tt.TreeTagger(TAGLANG='de', TAGDIR="C:\\TreeTagger")
  text = "Hallöchen Manfred, ich, der Titus, möchte dir mal was sagen: 'Ich find das Toll'. Manfred schaute betröppelt in die Röhre und konnte sein Glück kaum fassen. Schon am nächsten Morgen würde er in einem Flieger nach Hause fliegen."
  (words, tags, lemmas) = POS_tagger(tagger=t_tagger, document=text)
  word_len = word_length(document_word=words)
  syll_count = syllable_count(document_word=words)
  print(len(words), len(tags), len(lemmas), len(word_len), len(syll_count))

  print(pd.DataFrame(data={"Words": words, "Tags": tags, "Lemma": lemmas, "Word Length": word_len, "Syllabel count": syll_count}))

  print("Mean Word Length", 1/len(word_len) * sum(word_len))
  print("Mean Syll COunt", 1/len(word_len) * sum(syll_count))
  print("# Logicals: ", count_logicals(document_token=tags))
  print("type token ratio: ", type_token_ratio(document_token=tags))


  co_reference_matrix(document_tag=tags, document_lemma=lemmas)
  print(Flesch_Reading_Ease(document_words=words, document_tags=tags, document_syllables=syll_count))
  print(Flescher_Kincaid_Grad_Level(document_words=words, document_tags=tags, document_syllables=syll_count))

def POS_tagger(tagger, document):
  pos_tags = tagger.tag_text(document)
  words = [i.split("\t")[0] for i in pos_tags]
  tags = [i.split("\t")[1] for i in pos_tags]
  lemmas = [i.split("\t")[-1] for i in pos_tags]
  return (words, tags, lemmas)


def word_length(document_word):
  '''

  :param document: List of tokenized entries
  :return: length for each entry in the list
  '''
  return [len(i) for i in document_word]


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


# TODO:
def pronoun_resolution(document):
  # This Task, wants to map pronouns with their respectiv "owners"
  # more complex task!
  # Pronoun density  consists  of  the  proportion  of  noun  phrases(NPs, as defined  by  a  syntactic
  # parser, which  will  be  described  later) that  are  captured by  pronouns(as defined  by  the  Brill  POS  tagger).
  return None



# TODO:
def coreference_cohesion(document):
  ## Includes Pronoun resolution
  # Local and Global (1 and 2 meaning?)
  return None


# TODO:
def casual_cohesion(document):
  #??
  return None


def count_logicals(document_token):
  # high count means more "work"
  # KOUI:subordinating  conjunction  followed|  by \zu" and innitive um [zu leben], anstatt [zu fragen]
  # KOUS:  subordinating  conjunction  followed  by  clause  weil, dass, damit, wenn, ob
  # KON:  coordinating  conjunction  und, oder, aber
  # KOKOM:  comparative  conjunction  als, wie
  log_count = 0
  for i in document_token:
    if i in ["KOUI", "KOUS", "KON", "KOKOM"]:
      log_count += 1
  return log_count


def type_token_ratio(document_token):
  # count unique words against their repetitions.
  # split into Nouns and non-Noun content words
  counter_dict = {}
  counter_dict_non_Noun = {}
  for i in document_token:
    if i in ["NN", "NE"]:
      counter_dict[i] = counter_dict.get(i, 0) + 1
    elif i in ["ADJA", "ADJD", "ADV"] or i[0] =="V": #, "VVFIN, VVIMP", "VVINF", "VVIZU", "VVPP"]
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


def Flesch_Reading_Ease(document_words, document_tags, document_syllables):
  if len(document_words) < 200:
    return None
  num_sentences = document_tags.count("$.")
  ASL = len(document_words)/num_sentences # ratio #words/#sent
  ASW = sum(document_syllables)/len(document_words) # # ratio Syllables/Words
  return 206.835 - 1.015 * ASL - 84.6 * ASW

def Flescher_Kincaid_Grad_Level(document_words, document_tags, document_syllables):
  if len(document_words) < 200:
    return None
  num_sentences = document_tags.count("$.")
  ASL = len(document_words)/num_sentences # ratio #words/#sent
  ASW = sum(document_syllables)/len(document_words)
  return 0.39*ASL + 11.8*ASW - 15.59


 # TODO: Excluded are Noun Phrases (lack of implementation/knowledge)
def co_reference_matrix(document_tag, document_lemma):
  word_dict = {}
  sentence_count = 0
  for l, t in zip(document_lemma, document_tag):
    if t in ["NN", "NE"] or (t[0]== "P" and t[1]!= "T"):
      temp_dict = word_dict.get(l, [])
      temp_dict.append(sentence_count)
      word_dict[l] = temp_dict
    elif t =="$.":
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
  print(local_corefererence_cohesion, global_corefererence_cohesion, co_reference_dist_sum)
  print(co_reference_dist)
  print(co_reference_exists)





CohMatrix()



'''
# TODO:
###### CohMatrix Primary MEassures:
# LSA
# Reading Ease
#Reading rade Level
# Wordfrequency??
# NUmber of Words
# type-Token-Ratio (what effects does that have?)
# Connectives
# Distinct between types of texts

###########In Need of a database
##### Word characteristica:
# - Familiarity: How frequently a word appears in print.
#    Can be retrieved from some database
# - Concreteness: How concrete or nonabstract a word is, on the basis of human ratings.
# - Imageability: How easy it is to construct a mental image of the word in one’s mind, according to human ratings.
# - Colorado meaningfulness: These are the meaningfulness ratings from a corpus developed by Toglia and Battig (1978), multiplied by 100.
# - Paivio meaningfulness: This is the rated meaningfulness of the word, based on the norms of Paivio, Yuille, and  Madigan (1968) and Gilhooly and Logie (1980), multiplied by 100 to produce a range from 100 to 700.
# - Age of acquisition: This is the score of the age-ofacquisition norms (Gilhooly & Logie, 1980) multiplied by 100 to produce a range from 100 to 700. Age of acquisition captures the fact that some words appear in children’s language earlier than others.
#    Basically at what age one learns a set of words
# Those measures are then used to describe the whole text, a paragraph and a single sentence (e.g. mean, max)
# Needed is a "vast" database with just those words and ratings - within the scope and reproduceable fo other languages?

Hypernyms Level of abstractness/concretness (a word with many levels of abstraction is faster to interprete then one with fewer)
Polysemy: how many meanings does a word have
# requires some kind of database


# Connectives:
# (1) clarifying connectives, such as in other words and that is;
# (2) additive connectives, such as also and moreover;
# (3) temporal connectives, such as after, before, and when; and
# (4) causal connectives, such as because, so, and consequently.
# On another dimension, there is a contrast between positive
# and negative connectives. For example, adversative additive
# connectives (e.g., however, in contrast) and adversative
# causal connectives (e.g., although) are negative.



### Things to include:
# - Word frequency/Familarity
# - POS indices: How often a POS tag is used in a document (#of POS tag x/1000 -> seems random)
#   COmbine content words (Nouns, lexical verbs, adjectives and adverbs) and functional words (the rest?), and calculate this indice again
# - COunt Logical operators
# - WHat kind of a phrase is the current phrase?



# NP, VP: Verb oder Substantive des Satzes  ist Kopf des Satzes (wichtigester bestandteil) Willkür?!

# Causal cohesion
The total list of causal particles comes either from
this short list of verbs or from the causal conjunctions, transitional
adverbs, and causal connectives.
The current metric of causal cohesion, which is a primary
measure, is simply a ratio of causal particles (P) to
causal verbs (V).
# needed causal verbs and causal particles
How to derive them?

'''