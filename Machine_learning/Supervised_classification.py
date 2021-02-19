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
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import os
from Helper.Gutenberg_IDs import *


def supervised_ML(evaluation_label_path= os.path.join(os.getcwd(), "data", "Evaluation", "Evaluation_label.csv"),
                  new_document_path=os.path.join(os.getcwd(), "data", "score_collection_new_documents.tsv"),
                  extra_books_path=os.path.join(os.getcwd(), "data", "score_collection_extra_books.tsv"),
                  gutenberg_path=os.path.join(os.getcwd(), "data", "score_collection_selected_gutenberg.tsv"),
                  non_feature_list=["Gutenberg_id", "Title", "Author"],
                  quality_control_features=["Hitrate affective Scores", "Hitrate sentiment shift", "Vocabulary correlation"]):
    
    binary_label = "binary_label"
    continuos_label = "continuous_label"
    if not os.path.isdir(os.path.join(os.getcwd(), "data", "ML Results")):
        os.mkdir(os.path.join(os.getcwd(), "data", "ML Results"))


    # <editor-fold desc="Generate Datasets">
    df_evaluation_labels = pd.read_csv(evaluation_label_path)
    df_new_documents = pd.read_csv(new_document_path, sep="\t", encoding="ISO-8859-1")
    df_gutenberg = pd.read_csv(gutenberg_path, sep="\t", encoding="ISO-8859-1")
    df_extra_books = pd.read_csv(extra_books_path, sep="\t", encoding="ISO-8859-1")
    
    df_train = label_books(df_gutenberg=df_gutenberg, df_extra=df_extra_books,
                           extra_books_hard_reads=["Duerrenmatt - Die Physiker.txt"])
    df_evaluation = label_evaluation_data(df_eval=df_new_documents, df_evaluation_labeled=df_evaluation_labels)
    # </editor-fold>
    df_train = df_train.fillna(0)

    for k in ["All", "Oversampling", "Undersampling"]:
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
        for i in ["RandomForest", "SVM", "NaiveBayes"]:
            # for i in ["NaiveBayes"]:
            for j in ["en", "de"]:
                classifiers.append(
                    fit_classifier(df_temp, classifier=i, Language=j, mode=k, non_feature_list=[*non_feature_list, *quality_control_features, "Language"], binary_label=binary_label))
                # evaluate(model=classifiers[-1], model_name=i, Language=j, feature_list=complete_feature_list, mode=k)


def evaluate(model, Language, feature_list, model_name, mode):
    df_eval_pred = df_eval[feature_list]
    df_eval_pred = df_eval_pred.fillna(value=0)
    y_pred = model.predict(df_eval_pred)
    df_eval_corr = pd.DataFrame(data={"Labels": y_pred, "mean_label": df_eval["mean_label"]})
    sns.heatmap(data=df_eval_corr.corr(), cmap="YlGnBu", annot=True)
    plt.title("Correlation Heatmap for Language %s" % i)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "Classification_Correlation_%s_%s_%s.png" % (mode, model_name, Language)), dpi=400)
    plt.clf()
    


def normalize(df, columns):
    for column in columns:
        try:
            df[column] = df[column].abs() / df[column].abs().max()
        except:
            # print(column)
            pass
    return df


def general_statistics(df, mode, non_feature_list, binary_label):
    feature_list = [i for i in df.columns if i not in non_feature_list]

    # df = df[feature_list]
    for i in ["de", "en"]:
        df[df["Language"]==i].describe().to_csv(os.path.join(os.getcwd(), "data", "ML Results", "data_Description_for_%s_%s.tsv" % (i, mode)), sep="\t")
    
    # apply normalization techniques
    # df = normalize(df, df.columns)
    # print(df.dropna().shape[0], df.shape[0])
    # print([len(df[i]) for i in feature_list])
    # df_melt = pd.melt(df, id_vars=["Gutenberg_id", "Language"], value_vars=feature_list, var_name="scores", value_name="values")
    #
    #
    # sns.violinplot(x="values", y="scores",  orient="h", split=True, hue="Language", data=df_melt, hue_order=["en", "de"])
    # plt.suptitle("Data Distribution by Language and score")
    # plt.tight_layout()
    # plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_Normalized_value_distribution.png" % mode), dpi=400)
    # plt.clf()
    #
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


def oversampling(df, Languages=["en", "de"], classes=[0, 1]):
    df_both_Languages = None
    for i in Languages:
        df_new = None
        class_size = [df[(df["Language"]==i) & (df["label"]== j)].shape[0] for j in classes]
        for index_j, j in enumerate(class_size):
            for index_k, k in enumerate(class_size[index_j + 1:]):
                if j > k:
                    # print(index_j +  1 + index_k)
                    # print()
                    df_temp = df[(df["Language"]==i) & (df["label"]== classes[index_j +  1 + index_k])]
                    df_sample = df_temp.sample(n=j, replace=True)
                    if df_new is None:
                        df_stay = df[(df["Language"] == i) & (df["label"] == classes[index_j])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                elif j < k:
                    df_temp = df[(df["Language"]== i) & (df["label"]== classes[index_j])]
                    df_sample = df_temp.sample(n=k, replace=True)
                    if df_new is None:
                        df_stay = df[(df["Language"] == i) & (df["label"] == classes[index_j +  1 + index_k])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                else:
                    df_stay = df[(df["Language"] == i) & ((df["label"] == classes[index_j]) | (df["label"] == classes[index_j +  1 + index_k]))]
                    df_new = df_stay
        if df_both_Languages is None:
            df_both_Languages = df_new
        else:
            df_both_Languages = df_both_Languages.append(df_new, ignore_index=True)
    return df_both_Languages
    
    
def undersampling(df, Languages=["en", "de"], classes=[0, 1]):
    df_both_Languages = None
    for i in Languages:
        df_new = None
        class_size = [df[(df["Language"]==i) & (df["label"]== j)].shape[0] for j in classes]
        for index_j, j in enumerate(class_size):
            for index_k, k in enumerate(class_size[index_j + 1:]):
                if j < k:
                    df_temp = df[(df["Language"]==i) & (df["label"]== classes[index_j +  1 + index_k])]
                    df_sample = df_temp.sample(n=j, replace=True)
                    if df_new is None:
                        df_stay = df[(df["Language"] == i) & (df["label"] == classes[index_j])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                elif j > k:
                    df_temp = df[(df["Language"]== i) & (df["label"]== classes[index_j])]
                    df_sample = df_temp.sample(n=k, replace=True)
                    if df_new is None:
                        df_stay = df[(df["Language"] == i) & (df["label"] == classes[index_j +  1 + index_k])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                else:
                    df_stay = df[(df["Language"] == i) & ((df["label"] == classes[index_j]) | (df["label"] == classes[index_j +  1 + index_k]))]
                    df_new = df_stay

        if df_both_Languages is None:
            df_both_Languages = df_new
        else:
            df_both_Languages = df_both_Languages.append(df_new, ignore_index=True)
    return df_both_Languages


def fit_classifier(df, mode, classifier, Language, non_feature_list, binary_label):
    df = df[df["Language"] == Language]
    feature_list = [i for i in df.columns if i not in non_feature_list]
    X_data = df[feature_list]
    Y_data = df[binary_label]
    
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=42, shuffle=True)

    

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
    print("Model: ", classifier, "Language: ", Language, "\nAccuarcy: ", accuracy_score(y_test, y_pred), "\nF1 Score: ", f1_score(y_test, y_pred))
    return model


def plot_results(df, Language, classifier, mode):
    df = df.sort_values(by="model_impact", ascending=False)
    sns.barplot(y="features", x="model_impact", data=df, color="b", orient="h")

    plt.title("Feature importances for %s in %s" % (classifier, Language))
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_%s_Results_%s.png" % (mode, classifier, Language)), dpi=400)
    plt.clf()


def validation(model):
    external_data = pd.read_csv(os.path.join(os.getcwd(), "data", "score_collection_extra_books.tsv"),
                                delimiter='\t', index_col=0, encoding="ISO-8859-1")
    # external_data = pd.DataFrame(external_data)
    
    # external_data.rename(columns=df.iloc[0])
    
    
    # external_data = external_data.drop(["topic_overlap"], axis=1)
    external_data["label"] = external_data.apply(lambda x: 1 if x["title"] == "Duerrenmatt - Die Physiker.txt" else 0, axis=1)
    
    x_external = external_data[["mean_word_length","mean_syllables","count_logicals","count_conjugations","mean_sentence_length",
                 "mean_punctuations","mean_lexical_diversity","type_token_ratio_nouns","type_token_ratio_verbs",
                 "type_token_ratio_adverbs","type_token_ratio_adjectives","FRE","FKGL","count_repeated_words",
                 "num_word_repetitions","mean_concretness", "nouns_overlap","verbs_overlap",
                 "adverbs_overlap","adjectives_overlap", "sentiment_overlap"]]
    y_external = external_data["label"]
    y_pred = model.predict(x_external)
    print("Accuarcy: ", accuracy_score(y_external, y_pred), "\nF1 Score: ", f1_score(y_external, y_pred))
    print("Accuracy RandomForest: ", np.mean(y_external == y_pred))
    

def label_books(df_gutenberg, df_extra, extra_books_hard_reads=["Duerrenmatt - Die Physiker.txt"]):
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
    df_eval = df_eval.join(df_evaluation_labeled.set_index("identifier"), on="Title", rsuffix="_label")
    return df_eval



supervised_ML()