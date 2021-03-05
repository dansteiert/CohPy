import treetaggerwrapper as tt
import datetime
import os
import pandas as pd
import csv
from sys import getsizeof
## Import Own Functions:
from Helper.Helper_functions import load_score_df, load_word_freq, mean_of_list
from Helper.w2v_model import load_w2v
from Scoring_functions.Pipeline import pipeline
from Helper.Load_books import load_gutenberg, download_files



def main(Gutenberg_path = os.path.join(os.getcwd(), "data", "Gutenberg", "data.json"),
         Gutenberg_path_for_download = os.path.join(os.getcwd(), "data", "Gutenberg", "txt_files"),
         Treetagger_loc="C:\\TreeTagger", languages=("en", "de"),
         affective_score_paths=(os.path.join(os.getcwd(),"data", "Score files", "Twitter_SGNS_AffectiveSpace.rsc.csv"),
                                  os.path.join(os.getcwd(),"data", "Score files", "affective_norms.txt")),
         affective_score_separator=("\t", "\t"), affective_identifier=("WORD", "WORD"),
         affective_score_label=(["Anger", "Arousal", "Disgust", "Fear", "Happiness", "Joy", "Sadness", "Valency"],
                               ["Anger", "Arousal", "Disgust", "Fear", "Happiness", "Joy", "Sadness", "Valency"]),
         concreteness_score_label=("AbsConc", "AbstCon"),
         word_freq_path=(os.path.join(os.getcwd(),"data", "Score files", "eng_wikipedia_2016_1M-words.txt"),
                         os.path.join(os.getcwd(),"data", "Score files", "deu_wikipedia_2016_1M-words.txt")),
         word_freq_sep=("\t", "\t"), word_freq_index_col=(0, 0), word_freq_col=("frequency", "frequency"), word_freq_identifier=("word", "word"),
         word_freq_header=(None, None), word_freq_corpus_size=(1000000, 1000000),
         w2v_model_path=(os.path.join(os.getcwd(), "data", "Score files", "120sdewac_sg300.vec"),
                         os.path.join(os.getcwd(), "data", "Score files", "120sdewac_sg300.vec")),
         # w2v_model_path=(None, None),
         connective_path=(os.path.join(os.getcwd(), "data", "Score files", "Connectives_en.csv"),
                         os.path.join(os.getcwd(), "data", "Score files", "Connectives_de.csv")),
         connective_separator=(",", ","), connective_identifier=("WORD", "WORD"),
         connective_label=("Connective Type", "Connective Type"),
         run_Gutenberg=False, target_path_full_gutenberg=os.path.join(os.getcwd(), "data", "score_collection_full_gutenberg.tsv"),
         selected_Gutenberg=True, target_path_selected_gutenberg=os.path.join(os.getcwd(), "data", "score_collection_selected_gutenberg.tsv"),
         run_extra_books=False, target_path_extra_books=os.path.join(os.getcwd(), "data", "score_collection_extra_books.tsv"),
         file_path_extra_books=os.path.join(os.getcwd(), "data", "Extra_books"),
         run_new_documents=True, target_path_new_documents=os.path.join(os.getcwd(), "data", "score_collection_new_documents.tsv"),
         file_path_new_documents=os.path.join(os.getcwd(), "data", "New_documents")):
    """
    Download gutenberg project data, calculate for books from the gutenberg project (or a selection), a set of
    "extra books" and "new documents" a set of comprehensibility scores.
    The distinction between "extra books" and "new documents" is made, to account for additions to the guteberg project
    (extra books) and validation data, which got in the process the name "new documents".
    :param Gutenberg_path: Path to the gutenberg metadata file, or where it should be generated
    :param Gutenberg_path_for_download: Path to where the gutenberg project books are to be downloaded into/are stored
    :param Treetagger_loc: Location for the Treetager installation on your machine
    :param languages: 2 letter synonym, based on gutenberg corpus naming, english("en") and german("de") are implemented
     currently, subsequent changes are necessary to add new languages. language order is important since all other elements are depending on it!
    :param affective_score_paths: Path to the affective score files, in order of the language!
    :param affective_score_separator: argument passed to pandas read_csv as sep, for the affective score file, for each language
    :param affective_identifier: the identifier colum for the affective score, for each language
    :param affective_score_label: the names of the regarded affective scores as a list, for each language one
    :param concreteness_score_label: the label for the concreteness column (also within the affective score file!, str, one for each language
    :param word_freq_path: Path to the background corpus frequency file, one for each language
    :param word_freq_sep: argument passed to pandas read_csv function as sep, one for each language
    :param word_freq_index_col: argument passed to pandas read_csv function as index_col, , one for each language
    :param word_freq_col: column name in which the frequency is stored, one for each language
    :param word_freq_identifier: column name, in which the identifier(lemma/words) are stored, one for each language
    :param word_freq_header: argument passed to pandas read_csv function as header, one for each language
    :param word_freq_corpus_size: # of sentences within the background corpus, one for each language
    :param w2v_model_path: path to the sentiment w2v model, one for each language
    :param connective_path: path to the file with the connective words, one for each language
    :param connective_separator: argument passed to pandas read_csv function as sep, one for each language
    :param connective_identifier: column name of the identifier/word/lemma, one for each language
    :param connective_label: column name of the label of the type of connective, one for each language, one for each language
    :param run_Gutenberg: bool, whether the gutenberg corpus is downloaded and scores calculated for its books.
    :param target_path_full_gutenberg: path to a file, where the results are stored in for the gutenberg books
    :param selected_Gutenberg: bool, whether only a selection of books should be run through the pipeline
    :param target_path_selected_gutenberg: path to a file, where the results are stored in for the selected gutenberg books
    :param run_extra_books: bool, whether the files in "extra_books" are run through the pipeline
    :param target_path_extra_books: path to a file, where the results are stored in for the "extra books"
    :param file_path_extra_books: path to where the "extra books" are stored
    :param run_new_documents: bool, whether the files in "new_documents" are run through the pipeline
    :param target_path_new_documents: path to a file, where the results are stored in for the "new documents"
    :param file_path_new_documents: path to where the "new documents" are stored
    :return: None, results are writen to target_path_{{selected_}gutenberg, extra_books, new_document}
    """
    
    print(datetime.datetime.now(), "Start Readability Calculations")

    # <editor-fold desc="Load Dependencies">
    
    print(datetime.datetime.now(), "Load V2W model")
    # <editor-fold desc="Load W2V model">

    if len(set(w2v_model_path)) == len(w2v_model_path):
        w2v_model = load_w2v(w2v_model_path[0])
        w2v_model = [w2v_model, w2v_model]
    else:
        w2v_model = [load_w2v(path) for path in w2v_model_path]
    # </editor-fold>
    
    print(datetime.datetime.now(), "Load Tree Tagger")
    # <editor-fold desc="Load Tree Tagger modules">
    ## TreeTagger files need to be downloaded here: https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/
    # Tagsets can also be found on this page. Add them to the lib folder of TreeTagger
    tree_tagger = [tt.TreeTagger(TAGLANG=i, TAGDIR=Treetagger_loc) for i in languages]
    # </editor-fold>

    print(datetime.datetime.now(), "Load affective Scores")
    # <editor-fold desc="Load affective Scores">

    aff_conc_label =[[*affective_score_label[0], concreteness_score_label[0]],
                     [*affective_score_label[1], concreteness_score_label[1]]]
    df_affective_list = [load_score_df(path_to_file=path, sep=sep, column=label_list, identifier=identifier) for
                      path, sep, identifier, label_list in zip(affective_score_paths, affective_score_separator,
                                                               affective_identifier, aff_conc_label)]
    # df_affective_list = [None, None]
    # </editor-fold>

    print(datetime.datetime.now(), "Load Word Frequencies")
    # <editor-fold desc="Load Word Frequencies">
    df_background_corpus_freq_list = [load_word_freq(path=path, sep=sep, header=header, index_col=index_col, identifier=freq_ident,freq_column=freq_col) for
                       path, sep, header, index_col, freq_col, freq_ident in zip(word_freq_path, word_freq_sep, word_freq_header,
                                                                  word_freq_index_col, word_freq_col, word_freq_identifier)]
    # df_background_corpus_freq_list = [None, None]
    # </editor-fold>

    print(datetime.datetime.now(), "Load Connectives")
    # <editor-fold desc="Load Connectives">
    df_connective_list = [load_score_df(path_to_file=path, sep=sep, column=label, identifier=ident) for
                       path, sep, ident, label in zip(connective_path, connective_separator, connective_identifier, connective_label)]
    # </editor-fold>


    
    # </editor-fold>

  
    if run_extra_books:
        print(datetime.datetime.now(), "Readability calculations for books in Extra Books directory")
        
        # Run Books through Pipeline
        passed = [
            pipeline(text_path=os.path.join(file_path_extra_books, doc_language, file_name), language=doc_language,
                     language_order=languages,
                     w2v_model=w2v_model,
                     tagger=tree_tagger,
                     df_affective=df_affective_list, affective_score_label=affective_score_label,
                     concreteness_label=concreteness_score_label,
                     df_background_corpus_frequency=df_background_corpus_freq_list,
                     background_corpus_size=word_freq_corpus_size,
                     df_connective=df_connective_list, connective_type_label=connective_label,
                     title=file_name, author=None, gutenberg_id=None, target_path=target_path_extra_books,
                     gutenberg_meta_dict_elem=None)
            for doc_language in os.listdir(file_path_extra_books)
            for file_name in os.listdir(os.path.join(file_path_extra_books, doc_language))
            ]
        print("passed: ", mean_of_list(passed))

    if run_new_documents:
        print(datetime.datetime.now(), "Readability calculations for books in New Document directory")
    
        # Run Books through Pipeline
        passed = [
            pipeline(text_path=os.path.join(file_path_new_documents, doc_language, file_name), language=doc_language,
                     language_order=languages,
                     w2v_model=w2v_model,
                     tagger=tree_tagger,
                     df_affective=df_affective_list, affective_score_label=affective_score_label,
                     concreteness_label=concreteness_score_label,
                     df_background_corpus_frequency=df_background_corpus_freq_list,
                     background_corpus_size=word_freq_corpus_size,
                     df_connective=df_connective_list, connective_type_label=connective_label,
                     title=file_name, author=None, gutenberg_id=None, target_path=target_path_new_documents,
                     gutenberg_meta_dict_elem=None)
            for doc_language in os.listdir(file_path_new_documents)
            for file_name in os.listdir(os.path.join(file_path_new_documents, doc_language))
            ]
        print("passed: ", mean_of_list(passed))

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

        # <editor-fold desc="Prevent from book score recalculation">
        gutenberg_id = set([i["id"] for i in gutenberg_meta_data.get("books", [])])
        if selected_Gutenberg:
            target_path = target_path_selected_gutenberg
            from Helper.Gutenberg_IDs import ID_collection
            ID_collection = set(ID_collection)
            gutenberg_id = ID_collection
        else:
            target_path = target_path_full_gutenberg

        if os.path.isfile(target_path):
            df = pd.read_csv(target_path, sep="\t", usecols=["Gutenberg_id"])
            gutenberg_id = gutenberg_id.difference(set(df["Gutenberg_id"]))
        gutenberg_meta_data = [i for i in gutenberg_meta_data.get("books", []) if i["id"] in gutenberg_id]
        # </editor-fold>
        
        # Run Books through Pipeline
        passed = [pipeline(text_path=os.path.join(Gutenberg_path_for_download, str(meta_dict["id"]) + ".txt"),
                           language=None,
                           language_order=languages,
                           w2v_model=w2v_model,
                           tagger=tree_tagger,
                           df_affective=df_affective_list, affective_score_label=affective_score_label,
                           concreteness_label=concreteness_score_label,
                           df_background_corpus_frequency=df_background_corpus_freq_list,
                           background_corpus_size=word_freq_corpus_size,
                           df_connective=df_connective_list, connective_type_label=connective_label,
                           title=None, author=None, gutenberg_id=None,
                           target_path=target_path,
                           gutenberg_meta_dict_elem=meta_dict,
                           )
                  for meta_dict in gutenberg_meta_data
                  ]
        print("passed: ", mean_of_list(passed))
