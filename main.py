import treetaggerwrapper as tt
import os
import numpy as np
import pandas as pd

## Import Own Functions:
from Word_scorings import *
from COhMatrix_scorings import *
from Treetagger import POS_tagger
from model_preprocessing import *
from w2v_model import *
from LDA_model import *
from LSA_model import *
from Helper_functions import *


## enter the text:
from nltk.corpus import gutenberg
text = gutenberg.raw("carroll-alice.txt")


### choose the language of the text
select_language = "de"

## TreeTagger files need to be downloaded here: https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/
# Tagsets can also be found on this page. Add them to the lib folder of TreeTagger
t_tagger = tt.TreeTagger(TAGLANG=select_language, TAGDIR="C:\\TreeTagger")

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
    ## Adjectives:
    adjective_accept_tags = ["ADJA", "ADJD"]
    adjective_accept_tags_start_with = []
    adjective_exclude_tags = []
    adjective_exclude_tags_start_with = []
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
    ## Adjectives:
    adjective_accept_tags = []
    adjective_accept_tags_start_with = ["A"]
    adjective_exclude_tags = ["AV", "AT"]
    adjective_exclude_tags_start_with = []
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
    # </editor-fold>

(words, tags, lemmas) = POS_tagger(tagger=t_tagger, document=text)
w_by_sent = []
l_by_sent = []
temp_l = []

# how to get paragraphs??
for t, l in zip(tags, lemmas):
    if check_tags(tag=t, accept_tags=punctuation_fin_accept_tags, accept_tags_start_with=punctuation_fin_accept_tags_start_with,
                  exclude_tags=punctuation_fin_exclude_tags, exclude_tags_start_with=punctuation_fin_exclude_tags_start_with):
        l_by_sent.append(temp_l)
        temp_l = []
    else:
        temp_l.append(l)


word_len = word_length(document_word=words).mean()
syll_count = syllable_count(document_word=words).mean()
print(pd.DataFrame(
    data={"Words": words, "Tags": tags, "Lemma": lemmas, "Word Length": word_len, "Syllabel count": syll_count}))

print("Mean Word Length", 1 / len(word_len) * sum(word_len))
print("Mean Syll COunt", 1 / len(word_len) * sum(syll_count))
print("# Logicals: ", count_logicals(document_tags=tags, accept_tags=logical_accept_tags,
                                     accept_tags_start_with=logical_accept_tags_start_with, exclude_tags=logical_exclude_tags,
                                     exclude_tags_start_with=logical_exclude_tags_start_with))
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
for index, (l) in enumerate(l_by_sent):
    for l_2 in l_by_sent[index + 1:]:
        # print(l, w)
        # print(l_2, w_2)
        print("-----------------------")
        print(sentence_similarity(w2v=w2v_model, sent_a=l, sent_b=l_2))
        # print(sentence_similarity(w2v=w2v_model, sent_a=w, sent_b=w_2))


## concretness:
df_conc = load_score_file("data\\350k_ims_sorted copy.dat")
conc = []
for i in lemmas:
    temp = Concretness(lemma=i, df=df_conc)
    if temp is not None:
        conc.append(temp)
print(sum(conc), mean_of_list(conc))
print(variance_of_list(conc))


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


## DBS:
# MRC Psycholinguistic Database - can be watched and queried but not downloaded easily
# https://www.ldc.upenn.edu/language-resources/data/obtaining ## Needs Payment to access databases??
# WordNet Synonyms - Verb Synonyms; also Semantic overlap
# TACCO data: Say more and be more coherent: How text elaboration and cohesion can increase writing quality.
# Predicting math performance using natural language processing tools. In LAK ’17: Proceedings of the 7th International Learning Analytics and knowledge Conference: Understanding, informing and improvinglearning with data
# TASA not really found - http://lsa.colorado.edu/spaces.html - link to TASA does not work.

## get books from Gutenberg Project:
# http://self.gutenberg.org/CollectionCatalog.aspx
# ADd own Corpora - Where to find them in "plain text"? - DOes one has to create them by themselfes?
