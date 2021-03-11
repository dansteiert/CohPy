#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Feb  3 12:42:53 2021

@author: nils, daniel
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import os
from Helper.Gutenberg_IDs import *
import csv


def supervised_ML(evaluation_label_path= os.path.join(os.getcwd(), "data", "Evaluation", "Evaluation_label.csv"),
                  new_document_path=os.path.join(os.getcwd(), "data", "score_collection_new_documents.tsv"),
                  extra_books_path=os.path.join(os.getcwd(), "data", "score_collection_extra_books.tsv"),
                  gutenberg_path=os.path.join(os.getcwd(), "data", "score_collection_selected_gutenberg.tsv"),
                  non_feature_list=["Gutenberg_id", "Title", "Author"],
                  quality_control_features=["Hitrate affective Scores", "Hitrate sentiment shift", "Vocabulary correlation"]
                  ):
    """
    A work in progress approach to build a classifier for a binary classification problem.
    Including under and oversampling and the classifiers SVM, Random Forest, Nave Bayes and different test-train split sizes
    
    :param evaluation_label_path: path to labeled new document data
    :param new_document_path: path to the new_document score file
    :param extra_books_path: path to the extra_books score file
    :param gutenberg_path: path to the gutenberg data score file
    :param non_feature_list: a list of features not applicable to classification
    :param quality_control_features: a list of features, with the purpose of quality control
    :return:
    """
    binary_label = "binary_label"
    continuos_label = "continuous_label"
    if not os.path.isdir(os.path.join(os.getcwd(), "data", "ML Results")):
        os.mkdir(os.path.join(os.getcwd(), "data", "ML Results"))


    # <editor-fold desc="Generate Datasets">
    df_evaluation_labels = pd.read_csv(evaluation_label_path, usecols=["identifier", "continuous_label"])
    df_new_documents = pd.read_csv(new_document_path, sep="\t", encoding="ISO-8859-1")
    df_gutenberg = pd.read_csv(gutenberg_path, sep="\t", encoding="ISO-8859-1")
    df_extra_books = pd.read_csv(extra_books_path, sep="\t", encoding="ISO-8859-1")
    
    df_train = label_books(df_gutenberg=df_gutenberg, df_extra=df_extra_books,
                           extra_books_hard_reads=["Duerrenmatt - Die Physiker.txt"])
    df_evaluation = label_evaluation_data(df_eval=df_new_documents, df_evaluation_labeled=df_evaluation_labels)
    # </editor-fold>

    
    df_train = df_train.fillna(0)
    # for test_size in [0.3, 0.5]:
    for test_size in [0.3]:
        # for normalization in [True, False]:
        for normalization in [False]:
            if normalization:
                df_train = normalize(df_train, columns=df_train.columns)
    
            # for k in ["All", "Oversampling", "Undersampling"]:
            for k in ["Oversampling"]:
                print("Sampling Mode: ", k)
                if k == "All":
                    df_temp = df_train
                elif k == "Oversampling":
                    df_temp = oversampling(df_train)
                elif k == "Undersampling":
                    df_temp = undersampling(df_train)
                else:
                    df_temp = df_train
                    print("else case")
            
                df_temp.to_csv(os.path.join(os.getcwd(), "data", "ML Results", "%s_data.tsv" % k), sep="\t")
                general_statistics(df_temp, mode=k, non_feature_list=non_feature_list, binary_label=binary_label)
                classifiers = []
                # for i in ["RandomForest", "SVM", "NaiveBayes"]:
                for i in ["RandomForest"]:
                    for j in ["en", "de"]:
                        classifiers.append(
                            fit_classifier(df_temp, classifier=i, Language=j, non_feature_list=[*non_feature_list, *quality_control_features, "Language"],
                                           binary_label=binary_label, normalized=normalization, test_size=test_size, sampling_mode=k))
                        # evaluate(model=classifiers[-1], model_name=i, Language=j, feature_list=complete_feature_list, mode=k)



    

def normalize(df, columns):
    """
    Normalizes data to a range 0-1, based on the data range given in the data
    :param df: pandas dataframe of scores
    :param columns: the columns which need normalization
    :return: pandas dataframe with normalized scores
    """
    for column in columns:
        try:
            df[column] = df[column].abs() / df[column].abs().max()
        except:
            pass
    return df


def general_statistics(df, mode, binary_label):
    """
    Return the general statistics of the data
    :param df: pandas dataframe of scores
    :param mode: str, all data, under and oversampling
    :param binary_label: column for the binary label
    :return: None, generates a csv file with the statistic and two figures
    """
    for i in ["de", "en"]:
        df[df["Language"]==i].describe().to_csv(os.path.join(os.getcwd(), "data", "ML Results", "data_Description_for_%s_%s.tsv" % (i, mode)), sep="\t")
    
    sns.countplot(x=binary_label, data=df, hue="Language", hue_order=["en", "de"])
    plt.title("Data Distribution by Language and score Label")
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_Label_distribution.png" % mode), dpi=400)
    plt.clf()
    
    for i in ["en", "de"]:
        sns.heatmap(data=df[df["Language"]==i].corr(), cmap="YlGnBu")
        plt.title("Correlation Heatmap for Language %s" % i)
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_Feature_correlation_%s.png" % (mode, i)), dpi=400)
        plt.clf()


def oversampling(df, Languages=["en", "de"], classes=[0, 1], binary_label="binary_label"):
    """
    Oversample the data, for the minority group, separated for each language
    :param df: pandas dataframe with scores
    :param Languages: list of languages
    :param classes: the available classes
    :param binary_label: name of the binary_label column
    :return: pandas dataframe with oversampled data for each language
    """
    df_both_Languages = None
    for i in Languages:
        df_new = None
        class_size = [df[(df["Language"]==i) & (df[binary_label]== j)].shape[0] for j in classes]
        for index_j, j in enumerate(class_size):
            for index_k, k in enumerate(class_size[index_j + 1:]):
                if j > k:

                    df_temp = df[(df["Language"]==i) & (df[binary_label]== classes[index_j +  1 + index_k])]
                    df_sample = df_temp.sample(n=j, replace=True)
                    if df_new is None:
                        df_stay = df[(df["Language"] == i) & (df[binary_label] == classes[index_j])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                elif j < k:
                    df_temp = df[(df["Language"]== i) & (df[binary_label]== classes[index_j])]
                    df_sample = df_temp.sample(n=k, replace=True)
                    if df_new is None:
                        df_stay = df[(df["Language"] == i) & (df[binary_label] == classes[index_j +  1 + index_k])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                else:
                    df_stay = df[(df["Language"] == i) & ((df[binary_label] == classes[index_j]) | (df[binary_label] == classes[index_j +  1 + index_k]))]
                    df_new = df_stay
        if df_both_Languages is None:
            df_both_Languages = df_new
        else:
            df_both_Languages = df_both_Languages.append(df_new, ignore_index=True)
    return df_both_Languages
    
    
def undersampling(df, Languages=["en", "de"], classes=[0, 1], binary_label="binary_label"):
    """
    Undersample the data, for the majority class, separated for each language
    :param df: pandas dataframe with scores
    :param Languages: list of languages
    :param classes: the available classes
    :param binary_label: name of the binary_label column
    :return: pandas dataframe with undersampled data for each language
    """
    df_both_Languages = None
    for i in Languages:
        df_new = None
        class_size = [df[(df["Language"]==i) & (df[binary_label]== j)].shape[0] for j in classes]
        for index_j, j in enumerate(class_size):
            for index_k, k in enumerate(class_size[index_j + 1:]):
                if j < k:
                    df_temp = df[(df["Language"]==i) & (df[binary_label]== classes[index_j +  1 + index_k])]
                    df_sample = df_temp.sample(n=j, replace=True)
                    if df_new is None:
                        df_stay = df[(df["Language"] == i) & (df[binary_label] == classes[index_j])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                elif j > k:
                    df_temp = df[(df["Language"]== i) & (df[binary_label]== classes[index_j])]
                    df_sample = df_temp.sample(n=k, replace=True)
                    if df_new is None:
                        df_stay = df[(df["Language"] == i) & (df[binary_label] == classes[index_j +  1 + index_k])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                else:
                    df_stay = df[(df["Language"] == i) & ((df[binary_label] == classes[index_j]) | (df[binary_label] == classes[index_j +  1 + index_k]))]
                    df_new = df_stay

        if df_both_Languages is None:
            df_both_Languages = df_new
        else:
            df_both_Languages = df_both_Languages.append(df_new, ignore_index=True)
    return df_both_Languages


# TODO: Finish documentation - this file and Regression!!
def fit_classifier(df, classifier, Language, non_feature_list, binary_label, normalized, test_size, sampling_mode):
    """
    Fit a model to the given data in df, and return the fitted model
    :param df: pandas data frame with columns to predict
    :param classifier: str, name of the classifier
    :param Language: str, 2-Language code
    :param non_feature_list: list, of features, not to use in classification
    :param binary_label: str, label of the binary label column
    :param normalized: Bool, indicator whether the data has been normalized or not
    :param test_size: float, (0-1) to pass to  sklearns train_test_split function as test_size
    :param sampling_mode: str, of the sampling mode
    :return: fitted model
    """
    df = df[df["Language"] == Language]
    feature_list = [i for i in df.columns if i not in [*non_feature_list, binary_label]]
    X_data = df[feature_list]
    Y_data = df[binary_label]
    
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_size, random_state=42, shuffle=True)


    # test with y_test permuted for validity:
    # y_test = np.random.permutation(y_test)

    #Fit classifier and feature importance
    feature_names = X_data.columns

    if classifier == "RandomForest":
        # feature_names = ["mean_word_length","mean_syllables","count_logicals","count_conjugations","mean_sentence_length","mean_punctuations","mean_lexical_diversity","type_token_ratio_nouns","type_token_ratio_verbs","type_token_ratio_adverbs","type_token_ratio_adjectives","FRE","FKGL","count_repeated_words","num_word_repetitions","mean_concretness","hitrate_conc","nouns_overlap","verbs_overlap","adverbs_overlap","adjectives_overlap" ,"sentiment_overlap","sentiment_hitrate"]
        # Build a forest and compute the impurity-based feature importances
        model = ExtraTreesClassifier(n_estimators=250,
                                      random_state=0)
        
        model.fit(x_train, y_train)
        importances = model.feature_importances_
        sd = np.std([tree.feature_importances_ for tree in model.estimators_],
                     axis=0)
        df_model = pd.DataFrame(data={"features": feature_names, "model_impact": importances, "sd": sd})

    elif classifier == "SVM":
        model = svm.SVC(kernel='linear')
        model.fit(x_train, y_train)
        df_model = pd.DataFrame(data={"features": feature_names, "model_impact": model.coef_[0]})

    elif classifier == "NaiveBayes":
        model = GaussianNB()
        model.fit(x_train, y_train)
        # print(model.theta_)
        df_model = pd.DataFrame(data={"features": feature_names, "model_impact": model.theta_[0]})

    plot_results(df=df_model, Language=Language, classifier=classifier, mode=mode)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    result_dict = {"Model": classifier, "Language": Language, "Accuracy": accuracy_score(y_test, y_pred),
                   "Normalized": normalized, "Test size": test_size, "Sampling Mode": sampling_mode,
                   "F1 Score": f1_score(y_test, y_pred),"TP": cm[0][0], "TN": cm[0][1], "FN": cm[1][0], "FP": cm[1][1]}
    print(result_dict)
    
    return model


def plot_results(df, Language, classifier, mode):
    """
    Generate a plot with the "impact" of each label to the models labeling
    :param df: pandas dataframe, containing a column "model_impact" and "features"
    :param Language: str, 2 letter language code
    :param classifier: str, name of the classifier
    :param mode: str, name of the resampling method
    :return:None, a figure is created
    """
    df = df.sort_values(by="model_impact", ascending=False)
    # sns.barplot(y="features", x="model_impact", data=df, color="b", orient="h")
    sns.barplot(x="features", y="model_impact", data=df, color="b")

    plt.title("Feature importances for %s in %s" % (classifier, Language))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_%s_Results_%s.png" % (mode, classifier, Language)), dpi=400)
    plt.clf()


def label_books(df_gutenberg, df_extra, extra_books_hard_reads=["Duerrenmatt - Die Physiker.txt"]):
    """
    Labeling function for the binary classification, which is based on author reputation, who writes hard to read books and who writes easy to read books
    :param df_gutenberg: pd dataframe, results of the gutenberg project files
    :param df_extra: pd dataframe, results of the files in the "extra books" folder
    :param extra_books_hard_reads: list, titles of the hard to read books in the extra books folder
    :return: merged dataframe, with their labels given
    """
    df_extra["Gutenberg_id"] = df_extra.apply(lambda x: -2 if x["Title"] in extra_books_hard_reads else -1, axis=1)
    df = df_extra.append(df_gutenberg)
    hard_reads = [*James_joyce, *Leo_Tolstoy, *Moby_Dick, *Thomas_Mann, *Heinrich_von_Kleist, *J_W_v_Goethe, *Franz_Kafka,
                  *Non_gutenberg_ids_difficult]
    easy_reads = [*Grimm, *Alaeddin, *Arabian_nights, *Burrgoughs_Tarzan, *Wizard_of_Oz, *Defoe_Robinson_Crusoe, *Barrie_Peter_Pan,
                  *Mark_Twain, *Karl_May, *Robert_Louise_Stevenson, *Johanna_Spyri, *Emmy_von_Rhoden, *Carlo_Collodi,
                  *Lewis_Carroll, *Rudyard_Kipling,
                  *Heinrich_Hoffmann, *Grahame_Kenneth, *Bassewitz, *Arthur_Conan_Doyle, *Non_gutenberg_ids_easy]
    df["binary_label"] = df.apply(lambda x: 0 if x["Gutenberg_id"] in easy_reads else (1 if x["Gutenberg_id"] in hard_reads else None), axis=1)
    return df
  
    
def label_evaluation_data(df_eval, df_evaluation_labeled):
    """
    Join results and labels for the evaluation dataset
    :param df_eval: pd dataframe, results of the evaluation
    :param df_evaluation_labeled: pd dataframe, lables of the evaluation data
    :return: pd dataframe, join of the above
    """
    df_eval = df_eval.join(df_evaluation_labeled.set_index("identifier"), on="Title", rsuffix="_label")
    return df_eval


def ML_results_evaluation(target_path = os.path.join(os.getcwd(), "data", "ML Results", "ML_results.tsv"), scoring="F1 Score", test_size=0.3):
    """
    Generate a figure to compare the results of the machine learning approach
    :param scoring: str, name of the score function to look at
    :param test_size, float, the size earlier passed as test_size to sklearn train_test_split function
    :param target_path: Path to the ML results
    :return: None, Figure is generated
    """
    df = pd.read_csv(target_path, sep="\t")
    sns.catplot(x="Model", y=scoring, hue="Language", col="Normalized", row="Sampling Mode",
                sharey=True, sharex=True, data=df[df["Test size"]== test_size], kind="bar")
    plt.suptitle("%s - for different models - test size %f" % (scoring, test_size))
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_Test_size_%f.png" % ( scoring, test_size)), dpi=400)

