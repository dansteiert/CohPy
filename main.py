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



def main(Gutenberg_path = os.path.join(os.getcwd(), "data", "Gutenberg", "data.json"),
         Gutenberg_path_for_download = os.path.join(os.getcwd(), "data", "Gutenberg", "txt_files"),
         Treetagger_loc="C:\\TreeTagger", languages=("en", "de"),
         concretness_score_paths=(os.path.join(os.getcwd(),"data", "Twitter_SGNS_AffectiveSpace.rsc.csv"),
                                  os.path.join(os.getcwd(),"data", "affective_norms.txt")),
         concretness_score_separator=("\t", "\t"), concretness_score_word=("WORD", "WORD"),
         concretness_score_label=("AbsConc", "AbstCon"),
         run_Gutenberg=True, selected_Gutenberg=True, run_extra_books=False, run_new_document=False):
    print(datetime.datetime.now(), "program loaded")
    
    



    
    print(datetime.datetime.now(), "Load Tree Tagger")
    # <editor-fold desc="Load Tree Tagger modules">
    ## TreeTagger files need to be downloaded here: https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/
    # Tagsets can also be found on this page. Add them to the lib folder of TreeTagger
    tree_tagger = [tt.TreeTagger(TAGLANG=i, TAGDIR=Treetagger_loc) for i in languages]
    # </editor-fold>


    print(datetime.datetime.now(), "Load Concretness Scores")
    # <editor-fold desc="Load Concretness Scores">
    ## Concretness english:
    list_dict_conc = []
    for path, word, label, sep in zip(concretness_score_paths, concretness_score_word, concretness_score_label, concretness_score_separator):
        df_conc = load_score_file(path, sep=sep)
        list_dict_conc.append(list_to_dict(df=df_conc, column=label, identifier=word))
    # </editor-fold>

    print(datetime.datetime.now(), "Load V2W model")
    # <editor-fold desc="Load W2V model">
    # W2V Model
    ### Small Model
    # w2v_model = load_w2v("data\\250kGLEC_sg500.vec")
    ## Larger Model
    w2v_model = load_w2v("data\\120sdewac_sg300.vec")
    # w2v_model = None
    # </editor-fold>


    if run_extra_books:
        print(datetime.datetime.now(), "start loading non Gutenberg books")
        # <editor-fold desc="Create non Gutenberg books File collection">
        column_names = ["id", "gutenberg_id", "title", "author", "language", "mean_word_length", "mean_syllables",
                        "count_logicals", "count_conjugations", "mean_sentence_length", "mean_punctuations",
                        "mean_lexical_diversity", "type_token_ratio_nouns", "type_token_ratio_verbs",
                        "type_token_ratio_adverbs", "type_token_ratio_adjectives", "FRE", "FKGL", "count_repeated_words",
                        "num_word_repetitions", "mean_concretness", "hitrate_conc", "nouns_overlap",
                        "verbs_overlap", "adverbs_overlap", "adjectives_overlap", "sentiment_overlap", "sentiment_hitrate", "topic_overlap"]
        with open(os.path.join(os.getcwd(), "data", "score_collection_extra_books.tsv"), "w") as file:
            file.write("\t".join(column_names))
            file.write("\n")
        # </editor-fold>

        
        # <editor-fold desc="Readability calculations non - Gutenberg Books">
        for j in os.listdir(os.path.join(os.getcwd(), "data", "Extra_books")):
            for i in os.listdir(os.path.join(os.getcwd(), "data", "Extra_books", j)):
                try:
                    with open(os.path.join(os.getcwd(), "data", "Extra_books",j, i), "r", encoding="utf-8", errors="replace") as txt:
                        text = txt.readlines()
                        text = "".join(text)
                except:
                    print(i, "exception")
                    continue
    
                flipper = False
                for l, conc_dict, t_tagger in zip(languages, list_dict_conc, tree_tagger):
                    if j== l:
                        flipper = True
                        temp_list = pipeline(text=text, language=j, w2v_model=w2v_model, list_dict_conc=conc_dict, tagger=t_tagger)
                if not flipper:
                    print("language not yet implemented", j)
                    continue
                meta_list = [None, None, i, None, j]
                meta_list.extend(temp_list)
                meta_list = [str(j) for j in meta_list]
                
                with open(os.path.join(os.getcwd(), "data", "score_collection_extra_books.tsv"), "a") as file:
                    file.write("\t".join(meta_list))
                    file.write("\n")
        # </editor-fold>

    if run_Gutenberg:
        if not os.path.isfile(Gutenberg_path):
            print("Enter Path to metafile for Gutenberg Library - use gutenburg python package for retrieving")
    
        print(datetime.datetime.now(), "Load Gutenberg Meta data")
        # <editor-fold desc="Load Gutenberg corpus">
        gutenberg_meta_data = load_gutenberg(Gutenberg_path)
        if not os.path.isfile(
                os.path.join(Gutenberg_path_for_download, "%s.txt" % gutenberg_meta_data["books"][-1]["id"])):
            print(datetime.datetime.now(), "Download Gutenberg data - Non German IP needed!")
            download_files(data=gutenberg_meta_data, path_for_download=Gutenberg_path_for_download)
        gutenberg_books = gutenberg_meta_data["books"]
        # </editor-fold>
        
        
        # <editor-fold desc="Check for Gutenberg Books if a result collection exists">
        if not selected_Gutenberg:
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
                                "type_token_ratio_adverbs", "type_token_ratio_adjectives", "FRE", "FKGL",
                                "count_repeated_words",
                                "num_word_repetitions", "mean_concretness", "hitrate_conc", "nouns_overlap",
                                "verbs_overlap", "adverbs_overlap", "adjectives_overlap", "sentiment_overlap",
                                "sentiment_hitrate", "topic_overlap"]
                with open(os.path.join(os.getcwd(), "data", "score_collection.tsv"), "w") as file:
                    file.write("\t".join(column_names))
                    file.write("\n")
                max_index = 0
        # </editor-fold>
        
        
        if selected_Gutenberg:
            from Gutenberg_IDs import ID_collection
            ID_collection = sorted(list(set(ID_collection)))
            # column_names = ["id", "gutenberg_id", "title", "author", "language", "mean_word_length", "mean_syllables",
            #                 "count_logicals", "count_conjugations", "mean_sentence_length", "mean_punctuations",
            #                 "mean_lexical_diversity", "type_token_ratio_nouns", "type_token_ratio_verbs",
            #                 "type_token_ratio_adverbs", "type_token_ratio_adjectives", "FRE", "FKGL",
            #                 "count_repeated_words",
            #                 "num_word_repetitions", "mean_concretness", "hitrate_conc", "nouns_overlap",
            #                 "verbs_overlap", "adverbs_overlap", "adjectives_overlap", "sentiment_overlap",
            #                 "sentiment_hitrate", "topic_overlap"]
            # with open(os.path.join(os.getcwd(), "data", "score_collection_selected_ids.tsv"), "w") as file:
            #     file.write("\t".join(column_names))
            #     file.write("\n")
            current_pointer = 0
        
        print(datetime.datetime.now(), "start loading books (5% steps):")
        # <editor-fold desc="Readability calculation Gutenberg Books">
        size_gutenberg = len(gutenberg_books)
    
        
        for index, i in enumerate(gutenberg_books):
            if selected_Gutenberg:
                if i["id"] != ID_collection[current_pointer]:
                    continue
                else:
                    current_pointer += 1
            print(i["id"], "load_book")
            # <editor-fold desc="Get Metadata">
            if index % size_gutenberg == 0 and index !=0:
                print("#", end="")
            language = i["languages"][0]
            # Chose files for Concretness and the w2v model
            if language not in languages:
                print("language not yet implemented", language)
                continue
            # </editor-fold>
            
            
            # <editor-fold desc="Load Texts">
            try:
                if os.stat(os.path.join(Gutenberg_path_for_download, "%s.txt" % i["id"])).st_size >=2500000:
                    print("file too large", i["id"])
                    continue
                with open(os.path.join(Gutenberg_path_for_download, "%s.txt" % i["id"]), "r", errors="replace") as txt:
                    text = txt.readlines()
                    text = "".join(text)
            except:
                print("file cound not be loaded", i["id"])
                continue
            # </editor-fold>
    
    
            # <editor-fold desc="Run through Pipeline">
            flipper = False
            for l, conc_dict, t_tagger in zip(languages, list_dict_conc, tree_tagger):
                if language == l:
                    flipper = True
                    temp_list = pipeline(text=text, language=language, w2v_model=w2v_model, list_dict_conc=conc_dict,
                                         tagger=t_tagger)
            if not flipper:
                print("language not yet implemented", language)
                continue
            # </editor-fold>
    
    
            # <editor-fold desc="Write Results to File">
            try:
                title = i["title"]
                title = title.replace("\r", " ")
                title = title.replace("\n", " ")
                title = title
            except:
                title = None
            try:
                author = i["authors"][0]["name"]
            except:
                author = None
            try:
                if selected_Gutenberg:
                    meta_list = [index, i["id"], title, author, language]
                else:
                    meta_list = [index + max_index, i["id"], title, author, language]
            except:
                print("problem with metadata")
                continue
                
            meta_list.extend(temp_list)
            meta_list = [str(j) for j in meta_list]
            if selected_Gutenberg:
                with open(os.path.join(os.getcwd(), "data", "score_collection_selected_ids.tsv"), "a") as file:
                    file.write("\t".join(meta_list))
                    file.write("\n")
            else:
                with open(os.path.join(os.getcwd(), "data", "score_collection.tsv"), "a") as file:
                    file.write("\t".join(meta_list))
                    file.write("\n")
            # </editor-fold>
        # </editor-fold>
        
            




main()













