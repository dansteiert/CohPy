import treetaggerwrapper as tt
import datetime
import os
import pandas as pd
import csv
## Import Own Functions:
from Helper.Helper_functions import load_score_file, list_to_dict, load_word_freq
from Helper.w2v_model import load_w2v
from Scoring_functions.Pipeline import pipeline
from Helper.Load_books import load_gutenberg, download_files



def main(Gutenberg_path = os.path.join(os.getcwd(), "data", "Gutenberg", "data.json"),
         Gutenberg_path_for_download = os.path.join(os.getcwd(), "data", "Gutenberg", "txt_files"),
         Treetagger_loc="C:\\TreeTagger", languages=("en", "de"),
         affinity_score_paths=(os.path.join(os.getcwd(),"data", "Score files", "Twitter_SGNS_AffectiveSpace.rsc.csv"),
                                  os.path.join(os.getcwd(),"data", "Score files", "affective_norms.txt")),
         affinity_score_separator=("\t", "\t"), affinity_identifier=("WORD", "WORD"),
         affinity_score_label=(["Anger", "Arousal", "Disgust", "Fear", "Happiness", "Joy", "Sadness", "Valency"],
                               ["Anger", "Arousal", "Disgust", "Fear", "Happiness", "Joy", "Sadness", "Valency"]),
         concreteness_score_label=("AbsConc", "AbstCon"),
         word_freq_path=(os.path.join(os.getcwd(),"data", "Score files", "eng_wikipedia_2016_1M-words.txt"),
                         os.path.join(os.getcwd(),"data", "Score files", "deu_wikipedia_2016_1M-words.txt")),
         word_freq_sep=("\t", "\t"), word_freq_index_col=(0, 0), word_freq_col_names=(["word", "frequency"], ["word", "frequency"]),
         word_freq_header=(None, None), word_freq_corpus_size=(1000000, 1000000),
         w2v_model_path=(os.path.join(os.getcwd(), "data", "Score files", "120sdewac_sg300.vec"),
                         os.path.join(os.getcwd(), "data", "Score files", "120sdewac_sg300.vec")),
         connective_path=(os.path.join(os.getcwd(), "data", "Score files", "Connectives_en.csv"),
                         os.path.join(os.getcwd(), "data", "Score files", "Connectives_de.csv")),
         connective_separator=(",", ","), connective_identifier=("WORD", "WORD"),
         connective_label=("Connective Type", "Connective Type"),
         run_Gutenberg=False, target_path_full_gutenberg=os.path.join(os.getcwd(), "data", "score_collection_full_gutenberg.tsv"),
         selected_Gutenberg=False, target_path_selected_gutenberg=os.path.join(os.getcwd(), "data", "score_collection_selected_gutenberg.tsv"),
         run_extra_books=False, target_path_extra_books=os.path.join(os.getcwd(), "data", "score_collection_extra_books.tsv"),
         run_new_documents=True, target_path_new_documents=os.path.join(os.getcwd(), "data", "score_collection_new_documents.tsv")):
    print(datetime.datetime.now(), "program loaded")
    
    
    print(datetime.datetime.now(), "Load Tree Tagger")
    # <editor-fold desc="Load Tree Tagger modules">
    ## TreeTagger files need to be downloaded here: https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/
    # Tagsets can also be found on this page. Add them to the lib folder of TreeTagger
    tree_tagger = [tt.TreeTagger(TAGLANG=i, TAGDIR=Treetagger_loc) for i in languages]
    # </editor-fold>


    print(datetime.datetime.now(), "Load Affinity Scores")
    # <editor-fold desc="Load Affinity Scores">
    affinity_dicts = []
    concreteness_score_dicts = []

    for path, sep, identifier, label_list, conc_label in zip(affinity_score_paths, affinity_score_separator, affinity_identifier, affinity_score_label, concreteness_score_label):
        concreteness_score_dicts.append(list_to_dict(path_to_file=path, sep=sep, column=conc_label, identifier=identifier))
        affinity_dicts.append(list_to_dict(path_to_file=path, sep=sep, column=label_list, identifier=identifier))
    # </editor-fold>


    print(datetime.datetime.now(), "Load Word Frequencies")
    # <editor-fold desc="Load Word Frequencies">
    word_freq_dicts = [load_word_freq(path=path, sep=sep, header=header, index_col=index_col, names=names) for
                       path, sep, header, index_col, names in zip(word_freq_path, word_freq_sep, word_freq_header,
                                                                  word_freq_index_col, word_freq_col_names)]
    # </editor-fold>

    print(datetime.datetime.now(), "Load Connectives")
    # <editor-fold desc="Load Word Frequencies">
    connectives_dicts = [list_to_dict(path_to_file=path, sep=sep, column=label, identifier=ident) for
                       path, sep, ident, label in zip(connective_path, connective_separator, connective_identifier, connective_label)]
    # </editor-fold>


    print(datetime.datetime.now(), "Load V2W model")
    # <editor-fold desc="Load W2V model">
    if len(w2v_model_path) == 0:
        w2v_model = [None, None]
    elif len(set(w2v_model_path)) == len(w2v_model_path):
        w2v_model = load_w2v(w2v_model_path[0])
        w2v_model = [w2v_model, w2v_model]
    else:
        w2v_model = [load_w2v(path) for path in w2v_model_path]
    # </editor-fold>

    
    if run_new_documents:
        new_doc_path = os.path.join(os.getcwd(), "data", "new_documents")
        new_doc_languages = "en"
        for i in os.listdir(new_doc_path):
            try:
                with open(os.path.join(new_doc_path, i), "r", encoding="utf-8",
                          errors="replace") as txt:
                    text = txt.readlines()
                    text = "".join(text)
            except:
                print(i, "exception")
                continue
    
            flipper = False
            for t_tagger, l, aff_dict, aff_label, conc_dict, freq_dict, freq_corpus_size, conn_dict, neg_conn_name, pos_conn_name in zip(tree_tagger, languages,
                                                                   affinity_dicts, affinity_score_label,
                                                                   concreteness_score_dicts,
                                                                   word_freq_dicts, word_freq_corpus_size,
                                                                   connectives_dicts):
                if new_doc_languages == l:
                    flipper = True
                    temp_dict = pipeline(text=text, language=new_doc_languages, w2v_model=w2v_model, tagger=t_tagger,
                                         affinity_dict=aff_dict, affinity_score_label=aff_label, concreteness_dict=conc_dict,
                                         word_freq_dict=freq_dict, word_freq_corpus_size=freq_corpus_size,
                                         connective_dict=conn_dict)
            if not flipper:
                print("language not yet implemented", l)
                continue
            temp_dict = {**{"id": None, "Gutenberg_id": None, "Title": i, "Author": None, "Language": new_doc_languages}, **temp_dict}
            if os.path.isfile(target_path_new_documents):
                with open(os.path.join(os.getcwd(), "data", "score_collection_new_documents.tsv"), "a") as file:
                    writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in temp_dict.items()]), delimiter="\t")
                    for data in temp_dict:
                        writer.writerow(data)
            else:
                with open(os.path.join(os.getcwd(), "data", "score_collection_new_documents.tsv"), "w") as file:
                    writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in temp_dict.items()]), delimiter="\t")
                    writer.writeheader()
                    for data in temp_dict:
                        writer.writerow(data)

        # </editor-fold>
        
        

# TODO: new structure!!
    if run_extra_books:
        print(datetime.datetime.now(), "start loading non Gutenberg books")

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
            from Helper.Gutenberg_IDs import ID_collection
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













