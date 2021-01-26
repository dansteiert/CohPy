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
from Load_books import *
from Pipeline import *


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


def main(Gutenberg_path = os.path.join(os.getcwd(), "data", "Gutenberg", "data.json"), Gutenberg_path_for_download = os.path.join(os.getcwd(), "data", "Gutenberg", "txt_files")):
    print(datetime.datetime.now(), "program loaded")
    
    if not os.path.isfile(Gutenberg_path):
        print("Enter Path to metafile for Gutenberg Library - use gutenburg python package for retrieving")
    
    print(datetime.datetime.now(), "Load Gutenberg Meta data")
    gutenberg_meta_data = load_gutenberg(Gutenberg_path)
    # if not os.path.join(Gutenberg_path_for_download, "%s.txt" % gutenberg_meta_data["books"][-1]["id"]):
    #     print(datetime.datetime.now(), "Download Gutenberg data - Non German IP needed!")
    #     download_files(data=gutenberg_meta_data, path_for_download=Gutenberg_path_for_download)
    gutenberg_books = gutenberg_meta_data["books"]

    print(datetime.datetime.now(), "Load V2W model")
    # W2V Model
    ### Small Model
    # w2v_model = load_w2v("data\\250kGLEC_sg500.vec")
    ## Larger Model
    # w2v_model = load_w2v("data\\120sdewac_sg300.vec")
    w2v_model = None
    
    print(datetime.datetime.now(), "Load Tree Tagger")
    ## TreeTagger
    ## TreeTagger files need to be downloaded here: https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/
    # Tagsets can also be found on this page. Add them to the lib folder of TreeTagger
    t_tagger_en = tt.TreeTagger(TAGLANG="en", TAGDIR="C:\\TreeTagger")
    # t_tagger_de = tt.TreeTagger(TAGLANG="de", TAGDIR="C:\\TreeTagger")

    print(datetime.datetime.now(), "Load Concretness Scores")
    ## Concretness english:
    concretness_label = "AbsConc"
    word_label = "WORD"
    df_conc = load_score_file(os.path.join(os.getcwd(),"data", "Twitter_SGNS_AffectiveSpace.rsc.csv"), sep="\t")
    list_dict_conc_en = list_to_dict(df=df_conc, column=concretness_label, identifier=word_label)
    # ## Concretness german:
    # concretness_label = "AbstCon"
    # word_label = "WORD"
    # df_conc = load_score_file(os.path.join(os.getcwd(),"data", "affective_norms.txt"), sep="\t")
    # list_dict_conc_de = list_to_dict(df=df_conc, column=concretness_label, identifier=word_label)


    ## Check if some work was already done:
    if os.path.isfile(os.path.join(os.getcwd(), "data", "score_collection.tsv")):
        df = pd.read_csv(os.path.join(os.getcwd(), "data", "score_collection.tsv"), sep="\t")
        max_index = df["id"].max() + 1
        if max_index == max_index:
            gutenberg_books = gutenberg_books[max_index:]
            print(datetime.datetime.now(), max_index, "entries skipped")
    else:
        column_names = ["id", "gutenberg_id", "title", "author", "language", "mean_word_length", "mean_syllables",
                        "count_logicals", "count_conjugations", "mean_sentence_length", "mean_punctuations",
                        "mean_lexical_diversity", "type_token_ratio_nouns", "type_token_ratio_verbs",
                        "type_token_ratio_adverbs", "type_token_ratio_adjectives", "FRE", "FKGL", "count_repeated_words",
                        "num_word_repetitions", "mean_concretness", "hitrate_conc", "nouns_overlap",
                        "verbs_overlap", "adverbs_overlap", "adjectives_overlap"]
        with open(os.path.join(os.getcwd(), "data", "score_collection.tsv"), "w") as file:
            file.write("\t".join(column_names))
            file.write("\n")
        max_index = 0
    
    
    print(datetime.datetime.now(), "start loading books (5% steps):")
    size_gutenberg = len(gutenberg_books)

    dict_collection = []
    
    for index, i in enumerate(gutenberg_books):
        print(index + max_index)
        if index % size_gutenberg == 0 and index !=0:
            print("#", end="")
        language = i["languages"][0]
        # Chose files for Concretness and the w2v model
        if language not in ["en", "de"]:
            print("language not yet implemented", language)
            continue
        try:
            with open(os.path.join(Gutenberg_path_for_download, "%s.txt" % i["id"]), "r") as txt:
                text = txt.readlines()
                text = "".join(text)
        except:
            continue
        if language == "en":
            temp_list = pipeline(text=text, language=language, w2v_model=w2v_model, list_dict_conc=list_dict_conc_en, tagger=t_tagger_en)
        # elif language == "de":
        #     temp_list = pipeline(text=text, language=language, w2v_model=w2v_model, list_dict_conc=list_dict_conc_de, tagger=t_tagger_de)
        else:
            print("language not yet implemented", language)

        # meta_list = [index, i["id"], i["title"], i["authors"][0]["name"], language, i["categories"]]
        try:
            title = i["title"]
            title = title.replace("\r", " ")
            title = title.replace("\n", " ")
        except:
            title = None
        try:
            author = i["authors"][0]["name"]
        except:
            author = None
        try:
            meta_list = [index + max_index, i["id"], title, author, language]
        except:
            continue
            
        meta_list.extend(temp_list)
        meta_list = [str(j) for j in meta_list]
        # temp_list["language"] = language
        # temp_dict["title"] = i["title"]
        # temp_dict["author"] = i["authors"][0]["name"]
        # temp_dict["categories"] = i["categories"]
        # temp_dict["gutenberg_id"] = i["id"]
        # temp_dict["position_id"] = index
        # dict_collection.append(temp_dict)
        with open(os.path.join(os.getcwd(), "data", "score_collection.tsv"), "a") as file:
            file.write("\t".join(meta_list))
            file.write("\n")
        
            




main()













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