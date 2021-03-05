# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import keras
# from keras.models import Sequential
# from keras.layers import Dense  # Neural network
import numpy as np
from sklearn.linear_model import LinearRegression, ARDRegression, BayesianRidge, ElasticNet, Lasso
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression, SelectKBest, mutual_info_regression
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def continuous_prediction(target_path = os.path.join(os.getcwd(), "data", "ML Results", "Regression_data.csv")):
    df = pd.read_csv(target_path, delimiter=',', encoding="ISO-8859-1", index_col=0)
    # df = pd.DataFrame(df)
    
    df = df.replace(np.nan, -1)
    df = df.replace(np.inf, -1)
    df = df.fillna(-1)
    regression_analysis(df=df)
    

def regression_analysis(df, non_feature_list=["Gutenberg_id", "Title", "Author", "Language"],
                        quality_control_features = ["Hitrate affective Scores", "Hitrate sentiment shift", "Vocabulary correlation"],
                        continuous_label="continuous_label"):

    feature_list = [i for i in df.columns if i not in [*non_feature_list, *quality_control_features, continuous_label]]
    
    
    SWRF = df[df["Title"].str.startswith("SWRF-V_Rohdaten", na=False)]
    SWRF_Y = SWRF[continuous_label]
    SWRF_X = SWRF[feature_list]
    
    ZVV = df[df["Title"].str.startswith("Zweisatz_Vorstudie_Verstehen", na=False)]
    ZVV_Y = ZVV[continuous_label]
    ZVV_X = ZVV[feature_list]
    
    CS = df[df["Title"].str.startswith("English_novel_christie_styles_sents", na=False)]
    CS_Y = CS[continuous_label]
    CS_X = CS[feature_list]
    
    regressor_list = [LinearRegression(n_jobs=-1), ARDRegression(n_iter=50), BayesianRidge(n_iter=50), ElasticNet(), Lasso(),
                    AdaBoostRegressor(loss="square"), BaggingRegressor(n_jobs=-1), ExtraTreesRegressor(n_jobs=-1), GradientBoostingRegressor(),
                      RandomForestRegressor(n_jobs=-1)]
    regressor_names = ["LinearRegression", "BayesianRidge", "ElasticNet", "Lasso",
                    "AdaBoostRegressor", "BaggingRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor",
                      "RandomForestRegressor"]
    
    # for X_data, y_data, name in zip([SWRF_X, ZVV_X, CS_X], [SWRF_Y, ZVV_Y, CS_Y], ["SWRF", "ZVV", "CS"]):
    for X_data, y_data, name in zip([CS_X], [CS_Y], ["CS"]):
        print(name)
        Linear_regerssion(X=X_data, y=y_data, scoring_function=f_regression, scoring_function_name="f_regression",
                          num_features=len(feature_list),
                          regressor_name=regressor_names, regressor=regressor_list, dataset_name=name)
        # Linear_regerssion(X=X_data, y=y_data, scoring_function=mutual_info_regression, num_features=len(feature_list))


## Results SWRF 29 Feauters f_regression R2=0.496786837176466
## Results SWRF 45 Feauters mutual information R2=0.4923013086331605
## Results ZVV 49 Feauters f_regression R2=0.22875175933266922
## Results ZVV 22 Feauters mutual information R2=0.2808433360869045
## Results CS 13 Feauters f_regression R2=0.08615368667902612
## Results CS 30 Feauters mutual information R2=0.09038118817925167


def Linear_regerssion(X, y, num_features, regressor, dataset_name, regressor_name, scoring_function=mutual_info_regression,
                      scoring_function_name="mutual_information",
                      target_path = os.path.join(os.getcwd(), "data", "ML Results", "Regression_results.tsv"),
                      
):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    for k in range(num_features, 30, -5):
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






# %%

def model_trainer(X_data, Y_data, iteration):
    best_r2 = -100000000
    for i in range(0, iteration):
        
        model = Sequential()
        model.add(Dense(128, input_dim=X_data.shape[1], activation="relu"))
        model.add(Dense(64, input_dim=X_data.shape[1], activation="relu"))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss="mean_squared_error", optimizer='adam')
        
        history = model.fit(X_data, Y_data, epochs=100, batch_size=32, verbose=0)
        prediction = model.predict(X_data)
        
        current_r2 = r2_score(Y_data, prediction)
        
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_model = model
    
    return best_model, best_r2


# %%

# best_model, best_r2 = model_trainer(CS_X_data, CS_Y_data, 100)
# print(best_r2)
#
# # %%
#
# best_model, best_r2 = model_trainer(SWRF_X_data, SWRF_Y_data, 100)
# print(best_r2)
#
# # %%
#
# best_model, best_r2 = model_trainer(ZVV_X_data, ZVV_Y_data, 100)
# print(best_r2)
continuous_prediction()

def Regression_evaluation(target_path = os.path.join(os.getcwd(), "data", "ML Results", "Regression_results_backup.tsv")):
    df = pd.read_csv(target_path, sep="\t")
    df[["k", "score"]] = df[["k", "score"]].apply(pd.to_numeric, errors='coerce')
    g = sns.catplot(x="k", y="score", hue="Regressor", row="Dataset name",
                    sharey = False, sharex = True, data = df, kind = "bar", aspect = 4, height = 3,
                    legend_out = False, ci = None)
    g.add_legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "Regression_results.png"), dpi=400)
