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
                            fit_classifier(df_temp, classifier=i, Language=j, mode=k, non_feature_list=[*non_feature_list, *quality_control_features, "Language"],
                                           binary_label=binary_label, normalized=normalization, test_size=test_size, sampling_mode=k))
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


def oversampling(df, Languages=["en", "de"], classes=[0, 1], binary_label="binary_label"):
    df_both_Languages = None
    for i in Languages:
        df_new = None
        class_size = [df[(df["Language"]==i) & (df[binary_label]== j)].shape[0] for j in classes]
        for index_j, j in enumerate(class_size):
            for index_k, k in enumerate(class_size[index_j + 1:]):
                if j > k:
                    # print(index_j +  1 + index_k)
                    # print()
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


def fit_classifier(df, mode, classifier, Language, non_feature_list, binary_label, normalized, test_size, sampling_mode):
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
    
    # # <editor-fold desc="Write results to target file">
    # target_path = os.path.join(os.getcwd(), "data", "ML Results", "ML_results.tsv")
    # if os.path.isfile(target_path):
    #     with open(target_path, "a") as file:
    #         writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in result_dict.items()]),
    #                                 delimiter="\t",
    #                                 lineterminator="\n")
    #         writer.writerow(result_dict)
    # else:
    #     with open(target_path, "w") as file:
    #         writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in result_dict.items()]),
    #                                 delimiter="\t",
    #                                 lineterminator="\n")
    #         writer.writeheader()
    #         writer.writerow(result_dict)
    # # </editor-fold>
    return model


def plot_results(df, Language, classifier, mode):
    df = df.sort_values(by="model_impact", ascending=False)
    # sns.barplot(y="features", x="model_impact", data=df, color="b", orient="h")
    sns.barplot(x="features", y="model_impact", data=df, color="b")

    plt.title("Feature importances for %s in %s" % (classifier, Language))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_%s_Results_%s.png" % (mode, classifier, Language)), dpi=400)
    plt.clf()


def validation(model, binary_label):
    external_data = pd.read_csv(os.path.join(os.getcwd(), "data", "score_collection_extra_books.tsv"),
                                delimiter='\t', index_col=0, encoding="ISO-8859-1")
    # external_data = pd.DataFrame(external_data)
    
    # external_data.rename(columns=df.iloc[0])
    
    
    # external_data = external_data.drop(["topic_overlap"], axis=1)
    external_data[binary_label] = external_data.apply(lambda x: 1 if x["title"] == "Duerrenmatt - Die Physiker.txt" else 0, axis=1)
    
    x_external = external_data[["mean_word_length","mean_syllables","count_logicals","count_conjugations","mean_sentence_length",
                 "mean_punctuations","mean_lexical_diversity","type_token_ratio_nouns","type_token_ratio_verbs",
                 "type_token_ratio_adverbs","type_token_ratio_adjectives","FRE","FKGL","count_repeated_words",
                 "num_word_repetitions","mean_concretness", "nouns_overlap","verbs_overlap",
                 "adverbs_overlap","adjectives_overlap", "sentiment_overlap"]]
    y_external = external_data[binary_label]
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


def ML_results_evaluation(target_path = os.path.join(os.getcwd(), "data", "ML Results", "ML_results.tsv")):
    df = pd.read_csv(target_path, sep="\t")
    sns.catplot(x="Model", y="F1 Score", hue="Language", col="Normalized", row="Sampling Mode",
                sharey=True, sharex=True, data=df[df["Test size"]== 0.3], kind="bar")
    plt.suptitle("F1 Score - for different models - test size 0.3")
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "F1_Score_Test_size_0.3.png"), dpi=400)

supervised_ML()