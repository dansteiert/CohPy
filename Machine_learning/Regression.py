# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense  # Neural network
import numpy as np
from sklearn.linear_model import LinearRegression, ARDRegression, BayesianRidge, ElasticNet, Lasso
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression, SelectKBest, mutual_info_regression
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns


def regression_analysis(target_path = os.path.join(os.getcwd(), "data", "ML Results", "Regression_data.csv"),
                        non_feature_list=["Gutenberg_id", "Title", "Author", "Language"],
                        quality_control_features = ["Hitrate affective Scores", "Hitrate sentiment shift", "Vocabulary correlation"],
                        continuous_label="continuous_label"):
    """
    Run multiple regression for each dataframe
    :param target_path: path, to the Datasets
    :param non_feature_list: list, of a non features, excluding languages and quality control features
    :param quality_control_features: list, of the column names of the quality control features
    :param continuous_label: str, label of the continuous label column
    :return: None, Regression results are saved
    """
    
    # <editor-fold desc="Read in data">
    df = pd.read_csv(target_path, delimiter=',', encoding="ISO-8859-1", index_col=0)
    df = df.replace(np.nan, -1)
    df = df.replace(np.inf, -1)
    df = df.fillna(-1)
    # </editor-fold>
    
    feature_list = [i for i in df.columns if i not in [*non_feature_list, *quality_control_features, continuous_label]]


    # <editor-fold desc="Generate Datasets to Train and Test the model with">
    SWRF = df[df["Title"].str.startswith("SWRF-V_Rohdaten", na=False)]
    SWRF_Y = SWRF[continuous_label]
    SWRF_X = SWRF[feature_list]
    
    ZVV = df[df["Title"].str.startswith("Zweisatz_Vorstudie_Verstehen", na=False)]
    ZVV_Y = ZVV[continuous_label]
    ZVV_X = ZVV[feature_list]
    
    CS = df[df["Title"].str.startswith("English_novel_christie_styles_sents", na=False)]
    CS_Y = CS[continuous_label]
    CS_X = CS[feature_list]
    # </editor-fold>

    # <editor-fold desc="Generate Regressors">
    regressor_list = [LinearRegression(n_jobs=-1), ARDRegression(n_iter=50), BayesianRidge(n_iter=50), ElasticNet(), Lasso(),
                    AdaBoostRegressor(loss="square"), BaggingRegressor(n_jobs=-1), ExtraTreesRegressor(n_jobs=-1), GradientBoostingRegressor(),
                      RandomForestRegressor(n_jobs=-1)]
    regressor_names = ["LinearRegression", "BayesianRidge", "ElasticNet", "Lasso",
                    "AdaBoostRegressor", "BaggingRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor",
                      "RandomForestRegressor"]
    # </editor-fold>

    # <editor-fold desc="Run Regression on each dataset and the regressors">
    for X_data, y_data, name in zip([SWRF_X, ZVV_X, CS_X], [SWRF_Y, ZVV_Y, CS_Y], ["SWRF", "ZVV", "CS"]):
        Linear_regerssion(X=X_data, y=y_data, scoring_function=f_regression, scoring_function_name="f_regression",
                          num_features=len(feature_list),
                          regressor_name=regressor_names, regressor=regressor_list, dataset_name=name)
        # Linear_regerssion(X=X_data, y=y_data, scoring_function=mutual_info_regression, num_features=len(feature_list))
    # </editor-fold>

    Regression_evaluation(target_path = os.path.join(os.getcwd(), "data", "ML Results", "Regression_results.tsv"))


def Linear_regerssion(X, y, num_features, regressor, dataset_name, regressor_name, scoring_function=mutual_info_regression,
                      scoring_function_name="mutual_information",
                      target_path = os.path.join(os.getcwd(), "data", "ML Results", "Regression_results.tsv")):
    """
    Test and Train the regressors, based on the k best selected features (for all possible k), the results (as R^2) are written out to an extra file
    :param X: Features for ML
    :param y: Label for ML
    :param num_features: the count of the features
    :param regressor: list, of all regressor instance, which are used
    :param dataset_name: str, the name of the dataset
    :param regressor_name: list, names of the regressors used
    :param scoring_function: scoring function instance for the k best feature selection
    :param scoring_function_name: str, name of the scoring function
    :param target_path: path, where the results are saved to
    :return: None, results are written to a file specified in taregt_path
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    for k in range(1, num_features, 1):
        # Select k best features
        fs = SelectKBest(score_func=scoring_function, k=k)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        
        for reg, reg_name in zip(regressor, regressor_name):
            result_dict = {"k":str(k), "Regressor": reg_name, "Scoring function": scoring_function_name, "Dataset name": dataset_name}
            # train regressor and test the results
            reg = reg.fit(X_train_fs, y_train)
            result_dict["score"] = reg.score(X_test_fs, y_test)
            # <editor-fold desc="Write results to target file">
            if os.path.isfile(target_path):
                with open(target_path, "a", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in result_dict.items()]),
                                            delimiter="\t",
                                            lineterminator="\n")
                    writer.writerow(result_dict)
            else:
                with open(target_path, "w", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=sorted([k for k, v in result_dict.items()]),
                                            delimiter="\t",
                                            lineterminator="\n")
                    writer.writeheader()
                    writer.writerow(result_dict)
            # </editor-fold>





def Regression_evaluation(target_path = os.path.join(os.getcwd(), "data", "ML Results", "Regression_results_backup.tsv")):
    """
    Generate a figure of the results of the regression
    :param target_path: path to the regression result file
    :return: None, a figure is generated
    """
    df = pd.read_csv(target_path, sep="\t")
    df[["k", "score"]] = df[["k", "score"]].apply(pd.to_numeric, errors='coerce')
    g = sns.catplot(x="k", y="score", hue="Regressor", row="Dataset name",
                    sharey = False, sharex = True, data = df, kind = "bar", aspect = 4, height = 3,
                    legend_out = False, ci = None)
    g.add_legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "Regression_results.png"), dpi=400)
