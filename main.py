
# TODO:
#  GermaNet Licence: https://uni-tuebingen.de/en/faculties/faculty-of-humanities/departments/modern-languages/department-of-linguistics/chairs/general-and-computational-linguistics/ressources/lexica/germanet/licenses/
#  Is their already something like that available or a similar databse to use?

# TODO:
#  PosTagging:
#  - pattern library from CLiPS Research Center, which implements POS taggers for German, -> 84% accuracy
#  - TIGER corpus from the Institute for Natural Language Processing / University of Stuttgart -> build your own ML POS tagger
#   o https://datascience.blog.wzb.eu/2016/07/13/accurate-part-of-speech-tagging-of-german-texts-with-nltk/ -> explains how to do it
#  - TreeTagger Also Uni. Stuggart
#  DGD DB for tagged data
#  TIGER has also some data
#  AntTag + AntCorp for generating own data


# TODO:
#  Lammatization:
#   https://datascience.blog.wzb.eu/2017/05/19/lemmatization-of-german-language-text/ -> introduction of pattern.de module
#   https://pypi.org/project/germalemma/ -> library from the same author
#    https://github.com/jfilter/german-lemmatizer -> a further extension
import treetaggerwrapper as tt
from COhMatrix_scorings import *
from Treetagger import POS_tagger
from model_preprocessing import *
from w2v_model import *
from LDA_model import *
from LSA_model import *
import os
import numpy as np
import pandas as pd

t_tagger = tt.TreeTagger(TAGLANG='de', TAGDIR="C:\\TreeTagger")
text = "Hallöchen Manfred, ich, der Titus, möchte dir mal was sagen: 'Ich find das Toll'. Manfred schaute betröppelt in die Röhre und konnte sein Glück kaum fassen. Schon am nächsten Morgen würde er in einem Flieger nach Hause fliegen."
(words, tags, lemmas) = POS_tagger(tagger=t_tagger, document=text)
# print(words)
# print(tags)
# print(lemmas)
# # w_by_sent = []
# # t_by_sent = []
# # l_by_sent = []
# # for w, t, l in zip(words, tags, lemmas):
# #     if t == "$.":
# #         # w_by_sent.append(w)
# #         # t_by_sent.append(t)
# #         l_by_sent.append(l)



word_len = word_length(document_word=words).mean()
syll_count = syllable_count(document_word=words).mean()
# print(len(words), len(tags), len(lemmas), len(word_len), len(syll_count))
#
print(pd.DataFrame(
    data={"Words": words, "Tags": tags, "Lemma": lemmas, "Word Length": word_len, "Syllabel count": syll_count}))

print("Mean Word Length", 1 / len(word_len) * sum(word_len))
print("Mean Syll COunt", 1 / len(word_len) * sum(syll_count))
print("# Logicals: ", count_logicals(document_token=tags))
print("type token ratio: ", type_token_ratio(document_token=tags))

co_reference_matrix(document_tag=tags, document_lemma=lemmas)
print(Flescher_Reading_Ease(document_words=words, document_tags=tags, document_syllables=syll_count))
print(Flescher_Kincaid_Grade_Level(document_words=words, document_tags=tags, document_syllables=syll_count))


#
#
#
#
# ######## Model building
# # TODO: structure for document vs corpus
# dictionary, doc_freq_matrix, tfidf = preprocessing(corpus_tokens=lemmas)
# lsa_model = LSA(df_matrix=doc_freq_matrix, dictionary=dictionary)
# lda_model = LDA(df_matrix=doc_freq_matrix, dictionary=dictionary)


# w2v_model = load_w2v("data\\250kGLEC_sg500.vec")
# for index, i in enumerate(l_by_sent):
#     for j in l_by_sent[index:]:
#         print(sentence_similarity(w2v=w2v_model, sent_a=i, sent_b=j))
# TODO:
#  CohMatrix
#  o A database of lots of German or what ever other language, texts.
#       - Wordfrequency/Familarity of words
#  o A dictionary with many words and Scores for:
#       - Concrete/Absrtactness -> Hypernmys
#       - Ease of Imagability
#       - Meaningfullness ratings (not sure if they are applicable to German)
#       - Age of Aquisition (at what age does one usually learn those words
#       -Polysemy: how many meanings has a word
#  o How to retive the kind of sentence (or parts) one has NP/ VP
#  o set of causal verbs and particles -> causal cohesion
#  TACCO:
#  o DB of texts
#  generally extend code to n-grams!
#  adapt for multiple languages (as far as possible)
