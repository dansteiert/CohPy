import treetaggerwrapper as tt
import os
import numpy as np
import pandas as pd
import datetime

## Import Own Functions:
from Word_scorings import *
from COhMatrix_scorings import *
from Treetagger import POS_tagger
from model_preprocessing import *
from w2v_model import *
from LDA_model import *
from LSA_model import *
from Helper_functions import *
from Ratio_Scores import *
from Count_Scores import *
from Overlap_Scores import *

print(datetime.datetime.now(), "program loaded")


# TODO:
#  o give some text
#  o set the language parameter
#  o if using a different language then english or german, add a new tag set, also if the other english TreeTagger version is to be used!
#  o if a different POS Tagger is used, add a new tag set as well!
#  o Set path to appropriate Concretness Scores List
#  o Set path to appropriate W2V model



## Databases for evaluations:
# MRC Psycholinguistic Database - can be watched and queried but not downloaded easily
# https://www.ldc.upenn.edu/language-resources/data/obtaining ## Needs Payment to access databases??
# TACCO data: Say more and be more coherent: How text elaboration and cohesion can increase writing quality.
# Predicting math performance using natural language processing tools. In LAK ’17: Proceedings of the 7th International Learning Analytics and knowledge Conference: Understanding, informing and improvinglearning with data
# TASA not really found - http://lsa.colorado.edu/spaces.html - link to TASA does not work.
# von Tübbinger Gruppe!

# https://compstat-lmu.github.io/seminar_nlp_ss20/resources-and-benchmarks-for-nlp.html
## Reading Comprehension datasets - lots of them - try to answer question specific for a given text.



## get books from Gutenberg Project:
# http://self.gutenberg.org/CollectionCatalog.aspx
# ADd own Corpora - Where to find them in "plain text"? - DOes one has to create them by themselves?
# a good page to find them?

# WordNet Synonyms - Verb Synonyms; also Semantic overlap ->Include just for english

# https://github.com/joeworsh/ai-lit

# TODO: Booklist
#  Source Good reads:
##  Children Books:
#  o The very hungrey caterpillar /Raupe Nimmersatt
#  o The Rainbow Fish
#  o Winnie the Pooh
#  o Tom Sawyer
#  o Pippi Longstockings/ Pipi Langstrumpf
#  o The Chronicals of Narnia??
#  o THe little Prince
#  o Charlie and the Chocolate Factory
#  o The Hobbit
##  Intermediat Reads:
#  o Harry Potter 1-7
#  o Lord of the Rings
## Difficult Reads:
#  o Ulysses -James Joyce
#  o Finnigans Wake -James Joyce
#  o Moby Dick
#  o War and Peace - Leo Tolstory
#  o The Name of the Rose
#  o Buddenbrooks - Thomas Mann
#  o Doktor Faustus - Thomas Mann
#  o Die Hermannsschalcht  v. Kleist
#  o Die Marrquise von O  v. Kleist
#  o Katechismus der Deutschen  v. Kleist


# TODO: Implementation:
#  o Word Frequency ->w2v/concretness Scores?
#  o Topic Modelling with LDA Mallet (Ask Christoph and Bae)
#  o Verb Synonyms -> WordNet!!!
#  o Correfference Matrix -> Look it up again
#  o Causal Cohesion Score -> Look it up again
#  o POS_Frequency vs Lexical Diversity -> Look it up again
#  o Content_functional_ratio -> where to get functional and content words from?
#  o LDA Mallet  + Proper preprocessing
#  Word frequency - List von Art! - log_f; und syllben count


# TODO:
#  Idea - Calculate the Scores for every thing (english and german -> separet those) apply some clustering
#  and define for each cluster "how hard it is to read", by selecting a clouple data points per cluster.
#  Do a min word threshold per Document



## enter the text:
from nltk.corpus import gutenberg
text = gutenberg.raw("carroll-alice.txt")


### choose the language of the text
select_language = "en"

# Chose files for Concretness and the w2v model

## Concretness english:
concretness_label = "AbsConc"
word_label = "WORD"
df_conc = load_score_file("data\\Twitter_SGNS_AffectiveSpace.rsc.csv", sep="\t")


# ## Concretness german:
# concretness_label = "AbstCon"
# word_label = "WORD"
# df_conc = load_score_file("data\\affective_norms.txt", sep="\t")


# W2V Model
### Small Model
# w2v_model = load_w2v("data\\250kGLEC_sg500.vec")
## Larger Model
w2v_model = load_w2v("data\\120sdewac_sg300.vec")

## TreeTagger files need to be downloaded here: https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/
# Tagsets can also be found on this page. Add them to the lib folder of TreeTagger

if select_language == "de":
    # <editor-fold desc="German Tagset">
    ## Nouns:
    nouns_accept_tags = []
    nouns_accept_tags_start_with = ["N"]
    nouns_exclude_tags = []
    nouns_exclude_tags_start_with = []
    ## Pronouns:
    pronouns_accept_tags = []
    pronouns_accept_tags_start_with = ["P"]
    pronouns_exclude_tags = ["PTK"]
    pronouns_exclude_tags_start_with = []
    ## Verbs:
    verbs_accept_tags = []
    verbs_accept_tags_start_with = ["V"]
    verbs_exclude_tags = []
    verbs_exclude_tags_start_with = []
    ## Adverbs:
    adverbs_accept_tags = ["ADV"]
    adverbs_accept_tags_start_with = []
    adverbs_exclude_tags = []
    adverbs_exclude_tags_start_with = []
    ## adjectives:
    adjectives_accept_tags = ["ADJA", "ADJD"]
    adjectives_accept_tags_start_with = []
    adjectives_exclude_tags = []
    adjectives_exclude_tags_start_with = []
    ## Punctuation:
    punctuation_accept_tags = []
    punctuation_accept_tags_start_with = ["$"]
    punctuation_exclude_tags = []
    punctuation_exclude_tags_start_with = []
    ## Punctuation Sentence Finishing:
    punctuation_fin_accept_tags = ["$."]
    punctuation_fin_accept_tags_start_with = []
    punctuation_fin_exclude_tags = []
    punctuation_fin_exclude_tags_start_with = []
    ## Conjugations:
    conjugations_accept_tags = []
    conjugations_accept_tags_start_with = ["K"]
    conjugations_exclude_tags = []
    conjugations_exclude_tags_start_with = []
    ## Logicals:
    logical_accept_tags = ["KON", "KOKOM"]
    logical_accept_tags_start_with = []
    logical_exclude_tags = []
    logical_exclude_tags_start_with = []
    ## Word_count:
    count_accept_tags = ["AD"]
    count_accept_tags_start_with = []
    count_exclude_tags = []
    count_exclude_tags_start_with = ["A", "C", "I", "K", "P", "T", "$"]
    # </editor-fold>
elif select_language == "en":
    # <editor-fold desc="English Tagset">
    ## Tag set discription can be found at: http://www.natcorp.ox.ac.uk/docs/c5spec.html

    ## Nouns:
    nouns_accept_tags = []
    nouns_accept_tags_start_with = ["N"]
    nouns_exclude_tags = []
    nouns_exclude_tags_start_with = []
    ## Pronouns:
    pronouns_accept_tags = []
    pronouns_accept_tags_start_with = ["P"]
    pronouns_exclude_tags = ["PU", "PR", "PO"]
    pronouns_exclude_tags_start_with = []
    ## Verbs:
    verbs_accept_tags = []
    verbs_accept_tags_start_with = ["V"]
    verbs_exclude_tags = []
    verbs_exclude_tags_start_with = []
    ## Adverbs:
    adverbs_accept_tags = []
    adverbs_accept_tags_start_with = ["A"]
    adverbs_exclude_tags = ["AJ", "AT"]
    adverbs_exclude_tags_start_with = []
    ## adjectives:
    adjectives_accept_tags = []
    adjectives_accept_tags_start_with = ["A"]
    adjectives_exclude_tags = ["AV", "AT"]
    adjectives_exclude_tags_start_with = []
    ## Punctuation:
    punctuation_accept_tags = []
    punctuation_accept_tags_start_with = ["P", "S"]
    punctuation_exclude_tags = ["PR", "PO", "PN"]
    punctuation_exclude_tags_start_with = []
    ## Punctuation Sentence Finishing:
    punctuation_fin_accept_tags = []
    punctuation_fin_accept_tags_start_with = ["S"]
    punctuation_fin_exclude_tags = []
    punctuation_fin_exclude_tags_start_with = []
    ## Conjugations:
    conjugations_accept_tags = []
    conjugations_accept_tags_start_with = ["C"]
    conjugations_exclude_tags = ["CR"]
    conjugations_exclude_tags_start_with = []
    ## Logicals:
    logical_accept_tags = ["CJC"]
    logical_accept_tags_start_with = []
    logical_exclude_tags = []
    logical_exclude_tags_start_with = []
    ## Word_count:
    count_accept_tags = []
    count_accept_tags_start_with = []
    count_exclude_tags = ["AT0", "CRD",]
    count_exclude_tags_start_with = ["S", "P", "D", "I", "E", "T", "X", "Z"]
    # </editor-fold>
else:
    print("no fitting language found")
    exit(-1)
    # return -1

t_tagger = tt.TreeTagger(TAGLANG=select_language, TAGDIR="C:\\TreeTagger")

print(datetime.datetime.now(), "preprocessing started")
# <editor-fold desc="Preprocessing">
segmented = split_at_newline(text=text, sep="\n\n")
wtl = [POS_tagger(tagger=t_tagger, document=i) for i in segmented]
words_by_seg = [i[0] for i in wtl]
tags_by_seg = [i[1] for i in wtl]
lemma_by_seg = [i[2] for i in wtl]
# lemma_by_seg_noun = [i[2] for i in wtl if check_tags(tag=i[1], accept_tags=nouns_accept_tags,
#                                           accept_tags_start_with=nouns_accept_tags_start_with, exclude_tags=nouns_exclude_tags,
#                                           exclude_tags_start_with =nouns_exclude_tags_start_with)]
# lemma_by_seg_adverb = [i[2] for i in wtl if check_tags(tag=i[1], accept_tags=adverbs_accept_tags,
#                                           accept_tags_start_with=adverbs_accept_tags_start_with, exclude_tags=adverbs_exclude_tags,
#                                           exclude_tags_start_with =adverbs_exclude_tags_start_with)]
# lemma_by_seg_adjective = [i[2] for i in wtl if check_tags(tag=i[1], accept_tags=adjectives_accept_tags,
#                                           accept_tags_start_with=adjectives_accept_tags_start_with, exclude_tags=adjectives_exclude_tags,
#                                           exclude_tags_start_with =adjectives_exclude_tags_start_with)]
# lemma_by_seg_verb = [i[2] for i in wtl if check_tags(tag=i[1] , accept_tags=verbs_accept_tags,
#                                           accept_tags_start_with=verbs_accept_tags_start_with, exclude_tags=verbs_exclude_tags,
#                                           exclude_tags_start_with =verbs_exclude_tags_start_with)]
# lemma_by_seg_pronoun = [i[2] for i in wtl if check_tags(tag=i[1], accept_tags=pronouns_accept_tags,
#                                accept_tags_start_with=pronouns_accept_tags_start_with, exclude_tags=pronouns_exclude_tags,
#                                exclude_tags_start_with =pronouns_exclude_tags_start_with)]

words = [j for i in wtl for j in i[0]]
tags = [j for i in wtl for j in i[1]]
lemma = [j for i in wtl for j in i[2]]

lemma_by_sentence = split_into_sentences(aggregator_list=lemma, document_tags=tags, accept_tags=punctuation_fin_accept_tags,
                                         accept_tags_start_with=punctuation_fin_accept_tags_start_with,
                                         exclude_tags=punctuation_fin_exclude_tags,
                                         exclude_tags_start_with=punctuation_fin_exclude_tags_start_with)
tags_by_sentence = split_into_sentences(aggregator_list=tags, document_tags=tags, accept_tags=punctuation_fin_accept_tags,
                                        accept_tags_start_with=punctuation_fin_accept_tags_start_with,
                                        exclude_tags=punctuation_fin_exclude_tags,
                                        exclude_tags_start_with=punctuation_fin_exclude_tags_start_with)
# </editor-fold>


print(datetime.datetime.now(), "Count Scores started")
# <editor-fold desc="Count Scores">
mean_word_length = word_length(document_word=words)
syllables_list = syllable_count(document_words=words)
mean_syllables = mean_of_list(syllables_list)
count_logicals = count_tags(document_tags=tags, accept_tags=logical_accept_tags,
                            accept_tags_start_with=logical_accept_tags_start_with, exclude_tags=logical_exclude_tags,
                            exclude_tags_start_with=logical_exclude_tags_start_with)
count_conjugations = count_tags(document_tags=tags, accept_tags=conjugations_accept_tags,
                                accept_tags_start_with=conjugations_accept_tags_start_with, exclude_tags=conjugations_exclude_tags,
                                exclude_tags_start_with=conjugations_exclude_tags_start_with)
mean_sentence_length = mean_of_list([len(i) for i in lemma_by_sentence])
mean_punctuations = mean_of_list([count_tags(document_tags=i, accept_tags=punctuation_accept_tags,
                                             accept_tags_start_with=punctuation_accept_tags_start_with,
                                             exclude_tags=punctuation_exclude_tags,
                                             exclude_tags_start_with=punctuation_exclude_tags_start_with) for i in tags_by_sentence])
mean_lexical_diversity = mean_of_list([lexical_diversity(document_tags=i, accept_tags=[], accept_tags_start_with=[],
                                                         exclude_tags=punctuation_accept_tags,
                                                         exclude_tags_start_with=punctuation_accept_tags_start_with)
                                       for i in tags_by_sentence])
# print(lemma, tags,count_accept_tags,count_accept_tags_start_with,count_exclude_tags,count_exclude_tags_start_with, sep="\n")
(count_repeated_words, num_word_repetitions) = word_repetition(document_lemma=lemma, document_tags=tags,
                                                             accept_tags=count_accept_tags,
                                                             accept_tags_start_with=count_accept_tags_start_with,
                                                             exclude_tags=count_exclude_tags,
                                                             exclude_tags_start_with=count_exclude_tags_start_with)

# TODO: add POS-Frequency and Connective Words
# </editor-fold>


print(datetime.datetime.now(), "Ratio Scores started")
# <editor-fold desc="Ratio Scores">
type_token_ratio_nouns = type_token_ratio(document_tags=tags, accept_tags=nouns_accept_tags,
                                          accept_tags_start_with=nouns_accept_tags_start_with, exclude_tags=nouns_exclude_tags,
                                          exclude_tags_start_with =nouns_exclude_tags_start_with)
type_token_ratio_adverbs = type_token_ratio(document_tags=tags, accept_tags=adverbs_accept_tags,
                                            accept_tags_start_with=adverbs_accept_tags_start_with, exclude_tags=adverbs_exclude_tags,
                                            exclude_tags_start_with =adverbs_exclude_tags_start_with)
type_token_ratio_adjectives = type_token_ratio(document_tags=tags, accept_tags=adjectives_accept_tags,
                                               accept_tags_start_with=adjectives_accept_tags_start_with, exclude_tags=adjectives_exclude_tags,
                                               exclude_tags_start_with =adjectives_exclude_tags_start_with)
type_token_ratio_verbs = type_token_ratio(document_tags=tags, accept_tags=verbs_accept_tags,
                                          accept_tags_start_with=verbs_accept_tags_start_with, exclude_tags=verbs_exclude_tags,
                                          exclude_tags_start_with =verbs_exclude_tags_start_with)

pronoun_noun_ratio = pronoun_resolution(document_tags=tags, nouns_accept_tags=nouns_accept_tags,
                                        nouns_accept_tags_start_with=nouns_accept_tags_start_with,
                                        nouns_exclude_tags=nouns_exclude_tags, nouns_exclude_tags_start_with=nouns_exclude_tags_start_with,
                                        pronouns_accept_tags=pronouns_accept_tags, pronouns_accept_tags_start_with=pronouns_accept_tags_start_with,
                                        pronouns_exclude_tags=pronouns_exclude_tags, pronouns_exclude_tags_start_with=pronouns_exclude_tags_start_with)

# TODO: add content_functional_ratio
# </editor-fold>


print(datetime.datetime.now(), "Overlap Scores started")
# <editor-fold desc="Overlaps">
nouns_overlap = overlap_matrix(lemma_by_segment=lemma_by_seg, tags_by_segment=tags_by_seg, accept_tags=nouns_accept_tags,
                               accept_tags_start_with=nouns_accept_tags_start_with, exclude_tags=nouns_exclude_tags,
                               exclude_tags_start_with =nouns_exclude_tags_start_with)
## Not Sure about Pronouns
# print(datetime.datetime.now(), "Noun Overlap Scores finished")
# pronouns_overlap = overlap_matrix(lemma_by_segment=lemma_by_seg, tags_by_segment=tags_by_seg, accept_tags=pronouns_accept_tags,
#                                   accept_tags_start_with=pronouns_accept_tags_start_with, exclude_tags=pronouns_exclude_tags,
#                                   exclude_tags_start_with =pronouns_exclude_tags_start_with)
# print(datetime.datetime.now(), "Pronoun Overlap Scores finished")

adverbs_overlap = overlap_matrix(lemma_by_segment=lemma_by_seg, tags_by_segment=tags_by_seg, accept_tags=adverbs_accept_tags,
                                 accept_tags_start_with=adverbs_accept_tags_start_with, exclude_tags=adverbs_exclude_tags,
                                 exclude_tags_start_with =adverbs_exclude_tags_start_with)
# print(datetime.datetime.now(), "Adverb Overlap Scores finished")

adjectives_overlap = overlap_matrix(lemma_by_segment=lemma_by_seg, tags_by_segment=tags_by_seg, accept_tags=adjectives_accept_tags,
                                    accept_tags_start_with=adjectives_accept_tags_start_with, exclude_tags=adjectives_exclude_tags,
                                    exclude_tags_start_with =adjectives_exclude_tags_start_with)
# print(datetime.datetime.now(), "Adjective Overlap Scores finished")

verbs_overlap = overlap_matrix(lemma_by_segment=lemma_by_seg, tags_by_segment=tags_by_seg, accept_tags=verbs_accept_tags,
                               accept_tags_start_with=verbs_accept_tags_start_with, exclude_tags=verbs_exclude_tags,
                               exclude_tags_start_with =verbs_exclude_tags_start_with)
# print(datetime.datetime.now(), "Verb Overlap Scores finished")

# </editor-fold>


print(datetime.datetime.now(), "Concretness Score started")
# <editor-fold desc="Concretness Score">
list_dict_conc = list_to_dict(df=df_conc, column=concretness_label, identifier=word_label)
mean_concretness, hitrate_conc = mean_concretness(lemma=lemma, list_dict=list_dict_conc)

# </editor-fold>


print(datetime.datetime.now(), "Sentiment Overlap started")
# <editor-fold desc="Sentiment Overlap">
sentiment_overlap, sentiment_hitrate = overlap_matrix_sentiment(w2v_model=w2v_model, lemma_by_segment=lemma_by_seg,
                                             tags_by_segment=tags_by_seg,   accept_tags=nouns_accept_tags,
                                             accept_tags_start_with=nouns_accept_tags_start_with, exclude_tags=nouns_exclude_tags,
                                             exclude_tags_start_with =nouns_exclude_tags_start_with)
## TODO: w2v model does not load properly! (only the smaller one)
# </editor-fold>


print(datetime.datetime.now(), "Other Scores (3) started")
# <editor-fold desc="Other Scores">
# co_ref = co_reference_matrix(document_tag=tags, document_lemma=lemma)
FRE = Flescher_Reading_Ease(document_words=words, document_syllables=syllables_list, num_sentences=len(lemma_by_sentence))
FKGL = Flescher_Kincaid_Grade_Level(document_words=words, document_syllables=syllables_list, num_sentences=len(lemma_by_sentence))
# </editor-fold>


print(datetime.datetime.now(), "Topic Modeling started")
# <editor-fold desc="Topic Modeling">
# TODO: redo with LDA Mallet!
# ######## Model building
# # TODO: structure for document vs corpus
# dictionary, doc_freq_matrix, tfidf = preprocessing(corpus_tokens=lemmas)
# lsa_model = LSA(df_matrix=doc_freq_matrix, dictionary=dictionary)
# lda_model = LDA(df_matrix=doc_freq_matrix, dictionary=dictionary)
# </editor-fold>



print(datetime.datetime.now(), "Result output")
list_of_results = [mean_word_length, mean_syllables, count_logicals, count_conjugations, mean_sentence_length,
                   mean_punctuations, mean_lexical_diversity, type_token_ratio_nouns,
                   type_token_ratio_verbs, type_token_ratio_adverbs, type_token_ratio_adjectives,
                   FRE, FKGL, count_repeated_words, num_word_repetitions, mean_concretness, hitrate_conc,
                   nouns_overlap, verbs_overlap, adverbs_overlap, adjectives_overlap]

list_of_results_names = ["mean_word_length", "mean_syllables", "count_logicals", "count_conjugations", "mean_sentence_length",
                         "mean_punctuations", "mean_lexical_diversity", "type_token_ratio_nouns",
                         "type_token_ratio_verbs", "type_token_ratio_adverbs", "type_token_ratio_adjectives",
                         "FRE", "FKGL", "count_repeated_words", "num_word_repetitions", "mean_concretness", "hitrate_conc",
                         "nouns_overlap", "verbs_overlap", "adverbs_overlap", "adjectives_overlap"]

for name, value in zip(list_of_results_names, list_of_results):
    print("%s: %.4f" % (name, value))


# print("\n#####################\nMatrices\n##############################\n")
#
# list_of_matrices = [co_ref]
# list_of_matrices_names = ["co_ref"]
# for name, value in zip(list_of_matrices_names, list_of_matrices):
#     print("\n#####################\n%s\n##############################\n" % name)
#     print(value)


print(datetime.datetime.now(), "Finished")




## Results:
# ALICE IN WONDERLAND
# mean_word_length: 3.4070
# mean_syllables: 1.3940
# count_logicals: 1083.0000
# count_conjugations: 1826.0000
# mean_sentence_length: 19.8384
# mean_punctuations: 3.0532
# mean_lexical_diversity: 8.1732
# type_token_ratio_nouns: 0.0009
# type_token_ratio_verbs: 0.0040
# type_token_ratio_adverbs: 0.0012
# type_token_ratio_adjectives: 0.0023
# FRE: 94.6390
# FKGL: 5.2359
# count_repeated_words: 1192.0000
# num_word_repetitions: 15593.0000
# mean_concretness: 2.2330
# hitrate_conc: 0.3665
# nouns_overlap: 556.0000
# verbs_overlap: 1082.0000
# adverbs_overlap: 243.0000
# adjectives_overlap: 85.0000
# sentiment_overlap: 277.59969237446785
# sentiment_hitrate : 0.8322338076456883