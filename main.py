import treetaggerwrapper as tt
import datetime
import os
import pandas as pd
import csv
## Import Own Functions:
from Helper.Helper_functions import load_score_df, load_word_freq
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
         word_freq_sep=("\t", "\t"), word_freq_index_col=(0, 0), word_freq_col=("frequency", "frequency"), word_freq_identifier=("word", "word"),
         word_freq_header=(None, None), word_freq_corpus_size=(1000000, 1000000),
         # w2v_model_path=(os.path.join(os.getcwd(), "data", "Score files", "120sdewac_sg300.vec"),
         #                 os.path.join(os.getcwd(), "data", "Score files", "120sdewac_sg300.vec")),
         w2v_model_path=(),
         connective_path=(os.path.join(os.getcwd(), "data", "Score files", "Connectives_en.csv"),
                         os.path.join(os.getcwd(), "data", "Score files", "Connectives_de.csv")),
         connective_separator=(",", ","), connective_identifier=("WORD", "WORD"),
         connective_label=("Connective Type", "Connective Type"),
         run_Gutenberg=False, target_path_full_gutenberg=os.path.join(os.getcwd(), "data", "score_collection_full_gutenberg.tsv"),
         selected_Gutenberg=False, target_path_selected_gutenberg=os.path.join(os.getcwd(), "data", "score_collection_selected_gutenberg.tsv"),
         run_extra_books=True, target_path_extra_books=os.path.join(os.getcwd(), "data", "score_collection_extra_books.tsv"),
         file_path_extra_books=os.path.join(os.getcwd(), "data", "Extra_books"),
         run_new_documents=False, target_path_new_documents=os.path.join(os.getcwd(), "data", "score_collection_new_documents.tsv"),
         file_path_new_documents=os.path.join(os.getcwd(), "data", "New_documents")):
    
    print(datetime.datetime.now(), "Start Readability Calculations")

    # <editor-fold desc="Load Dependencies">
    print(datetime.datetime.now(), "Load Tree Tagger")
    # <editor-fold desc="Load Tree Tagger modules">
    ## TreeTagger files need to be downloaded here: https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/
    # Tagsets can also be found on this page. Add them to the lib folder of TreeTagger
    tree_tagger = [tt.TreeTagger(TAGLANG=i, TAGDIR=Treetagger_loc) for i in languages]
    # </editor-fold>

    print(datetime.datetime.now(), "Load Affinity Scores")
    # <editor-fold desc="Load Affinity Scores">

    aff_conc_label =[[*affinity_score_label[0], concreteness_score_label[0]],
                     [*affinity_score_label[1], concreteness_score_label[1]]]
    df_affinity_list = [load_score_df(path_to_file=path, sep=sep, column=label_list, identifier=identifier) for
                      path, sep, identifier, label_list in zip(affinity_score_paths, affinity_score_separator,
                                                               affinity_identifier, aff_conc_label)]
    # </editor-fold>

    print(datetime.datetime.now(), "Load Word Frequencies")
    # <editor-fold desc="Load Word Frequencies">
    df_background_corpus_freq_list = [load_word_freq(path=path, sep=sep, header=header, index_col=index_col, identifier=freq_ident,freq_column=freq_col) for
                       path, sep, header, index_col, freq_col, freq_ident in zip(word_freq_path, word_freq_sep, word_freq_header,
                                                                  word_freq_index_col, word_freq_col, word_freq_identifier)]
    # </editor-fold>

    print(datetime.datetime.now(), "Load Connectives")
    # <editor-fold desc="Load Connectives">
    df_connective_list = [load_score_df(path_to_file=path, sep=sep, column=label, identifier=ident) for
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
    # </editor-fold>

    if run_new_documents:
        print(datetime.datetime.now(), "Readability calculations for books in New Document directory")
        # <editor-fold desc="Readability calculations for books in New Document directory">
        for doc_language in os.listdir(file_path_new_documents):
            for file_name in os.listdir(os.path.join(file_path_new_documents, doc_language)):
                # <editor-fold desc="Load Textfile">
                try:
                    with open(os.path.join(file_path_new_documents, doc_language, file_name), "r",
                              encoding="utf-8", errors="replace") as txt:
                        text = txt.readlines()
                        text = "".join(text)
                except:
                    print(os.path.join(file_path_new_documents, doc_language, file_name), "exception")
                    continue
                # </editor-fold>

                # <editor-fold desc="Run text file thorugh Pipeline">
                flipper = False
                for w2v_mod, t_tagger, l, df_affinity, aff_label, conc_label, df_background_corpus, freq_corpus_size, df_connective, conn_type_label in zip(w2v_model, tree_tagger,
                                                                                                    languages,
                                                                                                    df_affinity_list,
                                                                                                    affinity_score_label,
                                                                                                    concreteness_score_label,
                                                                                                    df_background_corpus_freq_list,
                                                                                                    word_freq_corpus_size,
                                                                                                    df_connective_list, connective_label):
                    if doc_language == l:
                        flipper = True
                        print(os.path.join(file_path_new_documents, doc_language, file_name), "progress: ", end="")
                        temp_dict = pipeline(text=text, language=l, w2v_model=w2v_mod,
                                             tagger=t_tagger,
                                             df_affinity=df_affinity, affinity_score_label=aff_label,
                                             concreteness_label=conc_label,
                                             df_background_corpus_frequency=df_background_corpus, background_corpus_size=freq_corpus_size,
                                             df_connective=df_connective, connective_type_label=conn_type_label)
                if not flipper:
                    print("language not yet implemented", l)
                    continue

                if not temp_dict:
                    continue
                    
                temp_dict = {
                    **{"id": None, "Gutenberg_id": None, "Title": file_name, "Author": None, "Language": doc_language},
                    **temp_dict}
                # </editor-fold>
                
                # <editor-fold desc="Write results to target file">
                if os.path.isfile(target_path_new_documents):
                    with open(target_path_new_documents, "a") as file:
                        writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in temp_dict.items()]),
                                                delimiter="\t",
                                                lineterminator="\n")
                        writer.writerow(temp_dict)
                else:
                    with open(target_path_new_documents, "w") as file:
                        writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in temp_dict.items()]),
                                                delimiter="\t",
                                                lineterminator="\n")
                        writer.writeheader()
                        writer.writerow(temp_dict)
                # </editor-fold>

        # </editor-fold>
  
    if run_extra_books:
        print(datetime.datetime.now(), "Readability calculations for books in Extra Books directory")
        # <editor-fold desc="Readability calculations non - Gutenberg Books">
        for doc_language in os.listdir(file_path_extra_books):
            for file_name in os.listdir(os.path.join(file_path_extra_books, doc_language)):
                # <editor-fold desc="Load text file">
                try:
                    with open(os.path.join(file_path_extra_books, doc_language, file_name), "r", encoding="utf-8", errors="replace") as txt:
                        text = txt.readlines()
                        text = "".join(text)
                except:
                    print(os.path.join(file_path_extra_books, doc_language, file_name), "exception")
                    continue
                # </editor-fold>

                # <editor-fold desc="Run text file thorugh Pipeline">
                flipper = False
                for w2v_mod, t_tagger, l, df_affinity, aff_label, conc_label, df_background_corpus, freq_corpus_size, df_connective, conn_type_label in zip(w2v_model, tree_tagger,
                                                                                                    languages,
                                                                                                    df_affinity_list,
                                                                                                    affinity_score_label,
                                                                                                    concreteness_score_label,
                                                                                                    df_background_corpus_freq_list,
                                                                                                    word_freq_corpus_size,
                                                                                                    df_connective_list, connective_label):
                    if doc_language == l:
                        flipper = True
                        print(os.path.join(file_path_new_documents, doc_language, file_name), "progress: ", end="")
                        temp_dict = pipeline(text=text, language=l, w2v_model=w2v_mod,
                                             tagger=t_tagger,
                                             df_affinity=df_affinity, affinity_score_label=aff_label,
                                             concreteness_label=conc_label,
                                             df_background_corpus_frequency=df_background_corpus, background_corpus_size=freq_corpus_size,
                                             df_connective=df_connective, connective_type_label=conn_type_label)
                if not flipper:
                    print("language not yet implemented", l)
                    continue

                if not temp_dict:
                    continue
                temp_dict = {
                    **{"id": None, "Gutenberg_id": None, "Title": file_name, "Author": None, "Language": doc_language},
                    **temp_dict}
                # </editor-fold>
                
                # <editor-fold desc="Write results to target file">
                if os.path.isfile(target_path_extra_books):
                    with open(target_path_extra_books, "a") as file:
                        writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in temp_dict.items()]),
                                                delimiter="\t",
                                                lineterminator="\n")
                        writer.writerow(temp_dict)
                else:
                    with open(target_path_extra_books, "w") as file:
                        writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in temp_dict.items()]),
                                                delimiter="\t",
                                                lineterminator="\n")
                        writer.writeheader()
                        writer.writerow(temp_dict)
                # </editor-fold>

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

        # <editor-fold desc="Check for Gutenberg Book ID if a result collection exists">
        if not selected_Gutenberg:
            if os.path.isfile(target_path_full_gutenberg):
                df = pd.read_csv(target_path_full_gutenberg, sep="\t")
                max_index = df["id"].max() + 1
                if max_index == max_index:
                    gutenberg_books = gutenberg_books[max_index:]
                    print(datetime.datetime.now(), max_index, "entries skipped")
            else:
                max_index = 0
        # </editor-fold>

        if selected_Gutenberg:
            from Helper.Gutenberg_IDs import ID_collection
            ID_collection = sorted(list(set(ID_collection)))
            current_pointer = 0

        print(datetime.datetime.now(), "Readability calculations for books in Gutenberg Books (selection)")
        # <editor-fold desc="Readability calculation Gutenberg Books">
        size_gutenberg = len(gutenberg_books)

        if selected_Gutenberg:
            target_path = target_path_selected_gutenberg
        else:
            target_path = target_path_full_gutenberg


        for index, i in enumerate(gutenberg_books):
            try:
                gb_index = i["id"]
            except:
                continue
            # <editor-fold desc="Iterate over metadata until a selected file is found - selected Gutenberg only">
            if selected_Gutenberg:
                try:
                    if gb_index != ID_collection[current_pointer]:
                        continue
                    else:
                        current_pointer += 1
                except:
                    break
            # </editor-fold>
            
            print(gb_index, "load_book")
            # <editor-fold desc="Get Metadata">
            if index % size_gutenberg == 0 and index !=0:
                print("#", end="")
                
            doc_language = i["languages"][0]
            if doc_language not in languages:
                print("language not yet implemented", doc_language)
                continue
                
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
                    target_index = index
                else:
                    target_index = index + max_index
            except:
                target_index = None
            # </editor-fold>

            # <editor-fold desc="Load Texts, if it is not too large">
            try:
                if os.stat(os.path.join(Gutenberg_path_for_download, "%s.txt" % gb_index)).st_size >=2500000:
                    print("file too large", gb_index)
                    continue
                with open(os.path.join(Gutenberg_path_for_download, "%s.txt" % gb_index), "r", errors="replace") as txt:
                    text = txt.readlines()
                    text = "".join(text)
            except:
                print("file cound not be loaded", gb_index)
                continue
            # </editor-fold>

            # <editor-fold desc="Run through Pipeline">
            flipper = False
            for w2v_mod, t_tagger, l, df_affinity, aff_label, conc_label, df_background_corpus, freq_corpus_size, df_connective, conn_type_label in zip(
                    w2v_model, tree_tagger,
                    languages,
                    df_affinity_list,
                    affinity_score_label,
                    concreteness_score_label,
                    df_background_corpus_freq_list,
                    word_freq_corpus_size,
                    df_connective_list, connective_label):
                if doc_language == l:
                    flipper = True
                    print(title, doc_language, "progress: ", end="")
                    temp_dict = pipeline(text=text, language=l, w2v_model=w2v_mod,
                                         tagger=t_tagger,
                                         df_affinity=df_affinity, affinity_score_label=aff_label,
                                         concreteness_label=conc_label,
                                         df_background_corpus_frequency=df_background_corpus,
                                         background_corpus_size=freq_corpus_size,
                                         df_connective=df_connective, connective_type_label=conn_type_label)

            if not flipper:
                print("language not yet implemented", l)
                continue

            if not temp_dict:
                continue
            
            temp_dict = {
                **{"id": target_index, "Gutenberg_id": gb_index, "Title": title, "Author": author, "Language": doc_language},
                **temp_dict}

            # <editor-fold desc="Write out results">
            if os.path.isfile(target_path):
                with open(target_path, "a") as file:
                    writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in temp_dict.items()]),
                                            delimiter="\t",
                                            lineterminator="\n")
                    writer.writerow(temp_dict)
            else:
                with open(target_path, "w") as file:
                    writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in temp_dict.items()]),
                                            delimiter="\t",
                                            lineterminator="\n")
                    writer.writeheader()
                    writer.writerow(temp_dict)
            # </editor-fold>
            # </editor-fold>

        # </editor-fold>

main()













