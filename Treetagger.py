import pandas as pd
import treetaggerwrapper as tt
import os

t_tagger = tt.TreeTagger(TAGLANG='de', TAGDIR="C:\\TreeTagger")

pos_tags = t_tagger.tag_text("Hallo ich wir sind doch haben hat gehabt zurechter Zeiten gewand")

original = []
lemmas = []
tags = []
for t in pos_tags:
  original.append(t.split('\t')[0])
  tags.append(t.split('\t')[1])
  lemmas.append(t.split('\t')[-1])

Results = pd.DataFrame({'Original': original, 'Lemma': lemmas, 'Tags': tags})
print(Results)

def word_length(document):
  '''

  :param document: List of tokenized entries
  :return: length for each entry in the list
  '''
  return [len(i) for i in document]

def syllable_count(document):
  '''

  :param document: List of tokenized entries
  :return: Syllable count for each entry in the list
  '''
  syllable_list = []
  for i in document:
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

def pronoun_resolution(document):
  # This Task, wants to map pronouns with their respectiv "owners"
  # more complex task!
  # Pronoun density  consists  of  the  proportion  of  noun  phrases(NPs, as defined  by  a  syntactic
  # parser, which  will  be  described  later) that  are  captured by  pronouns(as defined  by  the  Brill  POS  tagger).
  return None


###### CohMatrix Primary MEassures:
# LSA
# Reading Ease
#Reading rade Level
# Wordfrequency??
# NUmber of Words
# type-Token-Ratio (what effects does that have?)
# Connectives
# Distinct between types of texts


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

# Connectives:
# (1) clarifying connectives, such as in other words and that is;
# (2) additive connectives, such as also and moreover;
# (3) temporal connectives, such as after, before, and when; and
# (4) causalconnectives, such as because, so, and consequently.
# On another dimension, there is a contrast between positive
# and negative connectives. For example, adversative additive
# connectives (e.g., however, in contrast) and adversative
# causal connectives (e.g., although) are negative.



### Things to include:
# - Word frequency/Familarity
# - POS indices: How often a POS tag is used in a document (#of POS tag x/1000 -> seems random)
#   COmbine content words (Nouns, lexical verbs, adjectives and adverbs) and functional words (the rest?), and calculate this indice again
# - COunt Logical operators

def coreference_cohesion(document):
  ## Includes Pronoun resolution
  # Local and Global (1 and 2 meaning?)
  return None

def casual_cohesion(document):
  #??
  return None


def count_logicals(document):
  # high count means more "work"
  # KOUI:subordinating  conjunction  followed|  by \zu" and innitive um [zu leben], anstatt [zu fragen]
  # KOUS:  subordinating  conjunction  followed  by  clause  weil, dass, damit, wenn, ob
  # KON:  coordinating  conjunction  und, oder, aber
  # KOKOM:  comparative  conjunction  als, wie
  log_count = 0
  for i in document:
    if i in ["KOUI", "KOUS", "KON", "KOKOM"]:
      log_count += 1
  return log_count


def type_token_ratio(document):
  # count unique words against their repetitions.
  # split into Nouns and non-Noun content words
  counter_dict = {}
  counter_dict_non_Noun = {}
  for i in document:
    if i in ["NN", "NE"]:
      counter_dict[i] = counter_dict.get(i, 0) + 1
    elif i in ["ADJA", "ADJD", "ADV"] or i[0] =="V": #, "VVFIN, VVIMP", "VVINF", "VVIZU", "VVPP"]
      counter_dict_non_Noun[i] = counter_dict_non_Noun.get(i, 0) + 1

  return (len(counter_dict)/sum(counter_dict.values()), \
         len(counter_dict_non_Noun)/sum(counter_dict_non_Noun.items()))


# Further Reading Page 6 Polysemy and Hypernym