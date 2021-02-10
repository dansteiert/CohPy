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
from Gutenberg_IDs import *


def run_Feature_generation():
    if not os.path.isdir(os.path.join(os.getcwd(), "data", "ML Results")):
        os.mkdir(os.path.join(os.getcwd(), "data", "ML Results"))
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "score_collection_selected_ids.tsv"), delimiter='\t',
                     index_col=0, encoding="ISO-8859-1")
    
    complete_feature_list = ["mean_word_length","mean_syllables","count_logicals","count_conjugations","mean_sentence_length",
                 "mean_punctuations","mean_lexical_diversity","type_token_ratio_nouns","type_token_ratio_verbs",
                 "type_token_ratio_adverbs","type_token_ratio_adjectives","FRE","FKGL","count_repeated_words",
                 "num_word_repetitions","mean_concretness", "nouns_overlap","verbs_overlap",
                 "adverbs_overlap","adjectives_overlap", "sentiment_overlap"]
    
    # selected_feature_list =

    df["label"] = pd.Series([-1] * df.shape[0], index=df.index, dtype="int8")
    df = label_data(df)
    
    for k in ["All", "Oversampling", "Undersampling"]:
        print("Sampling Mode: ", k)
        if k =="All":
            df_temp = df
        elif k == "Oversampling":
            df_temp = oversampling(df)
        elif k == "Undersampling":
            df_temp = undersampling(df)
        else:
            df_temp = df
            print("else case")
            
        df_temp.to_csv(os.path.join(os.getcwd(), "data", "ML Results", "%s_data.tsv" % k), sep="\t")
        general_statistics(df_temp, mode=k)
        classifiers = []
        for i in ["RandomForest", "SVM", "NaiveBayes"]:
        # for i in ["NaiveBayes"]:
            for j in ["en", "de"]:
                classifiers.append(fit_classifier(df_temp, classifier=i, language=j, mode=k))


def normalize(df, columns):
    for column in columns:
        try:
            df[column] = df[column] / df[column].abs().max()
        except:
            # print(column)
            pass
    return df


def general_statistics(df, mode):
    for i in ["de", "en"]:
        df[df["language"]==i].describe().to_csv(os.path.join(os.getcwd(), "data", "ML Results", "data_Description_for_%s_%s.tsv" % (i, mode)), sep="\t")
    # print(df.describe())
    
    # apply normalization techniques
    df = normalize(df, df.columns)
    
    df_melt = pd.melt(df, id_vars=["gutenberg_id", "language"], value_vars=["mean_word_length","mean_syllables","count_logicals","count_conjugations","mean_sentence_length",
                 "mean_punctuations","mean_lexical_diversity","type_token_ratio_nouns","type_token_ratio_verbs",
                 "type_token_ratio_adverbs","type_token_ratio_adjectives","FRE","FKGL","count_repeated_words",
                 "num_word_repetitions","mean_concretness", "nouns_overlap","verbs_overlap",
                 "adverbs_overlap","adjectives_overlap", "sentiment_overlap", "sentiment_hitrate",
                                                              "hitrate_conc"], var_name="scores", value_name="values")



        # for i in ["mean_word_length","mean_syllables","count_logicals","count_conjugations","mean_sentence_length",
    #              "mean_punctuations","mean_lexical_diversity","type_token_ratio_nouns","type_token_ratio_verbs",
    #              "type_token_ratio_adverbs","type_token_ratio_adjectives","FRE","FKGL","count_repeated_words",
    #              "num_word_repetitions","mean_concretness", "nouns_overlap","verbs_overlap",
    #              "adverbs_overlap","adjectives_overlap", "sentiment_overlap", "sentiment_hitrate",
    #           "hitrate_conc"]:
    # fig, ax = plt.subplots(figsize=(10, 20))
    sns.violinplot(x="values", y="scores",  orient="h", split=True, hue="language", data=df_melt, hue_order=["en", "de"])
    plt.suptitle("Data Distribution by language and score")
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_Normalized_value_distribution.png" % mode), dpi=400)
    plt.clf()
        
    sns.countplot(x="label", data=df, hue="language", hue_order=["en", "de"])
    plt.title("Data Distribution by language and score Label")
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_Label_distribution.png" % mode), dpi=400)
    plt.clf()
    
    for i in ["en", "de"]:
        sns.heatmap(data=df[df["language"]==i].corr(), cmap="YlGnBu")
        plt.title("Correlation Heatmap for language %s" % i)
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_Feature_correlation_%s.png" % (mode, i)), dpi=400)
        plt.clf()


    
def label_data(df):
    # Im folgenden gleicht label = 1 schwer zu lesen und label = 0 leicht zu lesen.
    for j in [James_joyce, Leo_Tolstoy, Moby_Dick, Thomas_Mann, Heinrich_von_Kleist, J_W_v_Goethe, Franz_Kafka, Non_gutenberg_ids_difficult]:
        for index in j:
            row = np.flatnonzero(df["gutenberg_id"] == index)
            df.iloc[row, df.columns.get_loc('label')] = 1
    
    for j in [Grimm, Alaeddin, Arabian_nights, Burrgoughs_Tarzan, Wizard_of_Oz, Defoe_Robinson_Crusoe, Barrie_Peter_Pan,
              Mark_Twain, Karl_May, Robert_Louise_Stevenson, Johanna_Spyri, Emmy_von_Rhoden, Carlo_Collodi, Lewis_Carroll, Rudyard_Kipling,
              Heinrich_Hoffmann, Grahame_Kenneth, Bassewitz, Arthur_Conan_Doyle, Non_gutenberg_ids_easy]:
        for index in j:
            row = np.flatnonzero(df["gutenberg_id"] == index)
            df.iloc[row, df.columns.get_loc('label')] = 0
    # print(df["label"].tolist())
    return df
    
# Prepare training and test data

def oversampling(df, languages=["en", "de"], classes=[0, 1]):
    df_both_languages = None
    for i in languages:
        df_new = None
        class_size = [df[(df["language"]==i) & (df["label"]== j)].shape[0] for j in classes]
        for index_j, j in enumerate(class_size):
            for index_k, k in enumerate(class_size[index_j + 1:]):
                if j > k:
                    # print(index_j +  1 + index_k)
                    # print()
                    df_temp = df[(df["language"]==i) & (df["label"]== classes[index_j +  1 + index_k])]
                    df_sample = df_temp.sample(n=j, replace=True)
                    if df_new is None:
                        df_stay = df[(df["language"] == i) & (df["label"] == classes[index_j])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                elif j < k:
                    df_temp = df[(df["language"]== i) & (df["label"]== classes[index_j])]
                    df_sample = df_temp.sample(n=k, replace=True)
                    if df_new is None:
                        df_stay = df[(df["language"] == i) & (df["label"] == classes[index_j +  1 + index_k])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                else:
                    df_stay = df[(df["language"] == i) & ((df["label"] == classes[index_j]) | (df["label"] == classes[index_j +  1 + index_k]))]
                    df_new = df_stay
        if df_both_languages is None:
            df_both_languages = df_new
        else:
            df_both_languages = df_both_languages.append(df_new, ignore_index=True)
    return df_both_languages
    
def undersampling(df, languages=["en", "de"], classes=[0, 1]):
    df_both_languages = None
    for i in languages:
        df_new = None
        class_size = [df[(df["language"]==i) & (df["label"]== j)].shape[0] for j in classes]
        for index_j, j in enumerate(class_size):
            for index_k, k in enumerate(class_size[index_j + 1:]):
                if j < k:
                    df_temp = df[(df["language"]==i) & (df["label"]== classes[index_j +  1 + index_k])]
                    df_sample = df_temp.sample(n=j, replace=True)
                    if df_new is None:
                        df_stay = df[(df["language"] == i) & (df["label"] == classes[index_j])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                elif j > k:
                    df_temp = df[(df["language"]== i) & (df["label"]== classes[index_j])]
                    df_sample = df_temp.sample(n=k, replace=True)
                    if df_new is None:
                        df_stay = df[(df["language"] == i) & (df["label"] == classes[index_j +  1 + index_k])]
                        df_new = df_sample.append(df_stay, ignore_index=True)
                    else:
                        df_new = df_new.append(df_sample, ignore_index=True)
                else:
                    df_stay = df[(df["language"] == i) & ((df["label"] == classes[index_j]) | (df["label"] == classes[index_j +  1 + index_k]))]
                    df_new = df_stay

        if df_both_languages is None:
            df_both_languages = df_new
        else:
            df_both_languages = df_both_languages.append(df_new, ignore_index=True)
    return df_both_languages

def fit_classifier(df, mode, classifier="RandomForest", language="en",
                   feature_list=["mean_word_length", "count_conjugations", "mean_sentence_length",
                                 "type_token_ratio_nouns", "num_word_repetitions", "nouns_overlap", "mean_concretness"]):
    df = df[df["language"] == language]
    X_data = df[feature_list]
    Y_data = df["label"]
    
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
        # print(df_model)
        # indices = np.argsort(importances)[::-1]
        #
        # sorted_feature_list = []
        # for f in range(x_train.shape[1]):
        #     sorted_feature_list.append(feature_names[indices[f]])
        #
        # # Print the feature ranking
        # print("Feature ranking:")
        #
        # for f in range(x_train.shape[1]):
        #     print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

    elif classifier == "SVM":
        model = svm.SVC(kernel='linear')
        model.fit(x_train, y_train)
        df_model = pd.DataFrame(data={"features": feature_names, "model_impact": model.coef_[0]})

    elif classifier == "NaiveBayes":
        model = GaussianNB()
        model.fit(x_train, y_train)
        # print(model.theta_)
        df_model = pd.DataFrame(data={"features": feature_names, "model_impact": model.theta_[0]})

    plot_results(df=df_model, language=language, classifier=classifier, mode=mode)
    y_pred = model.predict(x_test)
    print("Model: ", classifier, "Language: ", language, "\nAccuarcy: ", accuracy_score(y_test, y_pred), "\nF1 Score: ", f1_score(y_test, y_pred))
    return model

def plot_results(df, language, classifier, mode):
    df = df.sort_values(by="model_impact", ascending=False)
    sns.barplot(y="features", x="model_impact", data=df, color="b", orient="h")

    plt.title("Feature importances for %s in %s" % (classifier, language))
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_%s_Results_%s.png" % (mode, classifier, language)), dpi=400)
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
    
run_Feature_generation()