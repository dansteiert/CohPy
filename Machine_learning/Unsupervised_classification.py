import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import os
from sklearn.manifold import TSNE


def run_clustering():
    """
    A work in progress script, with the aim to find cluster of closely related books, to be able to assign them a characteristic or even readability.
    :return:
    """
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "score_collection_selected_ids.tsv"), delimiter='\t',
                     index_col=0, encoding="ISO-8859-1")
    selected_features = ["language", "mean_word_length", "count_conjugations", "mean_sentence_length",
                        "type_token_ratio_nouns", "num_word_repetitions", "nouns_overlap", "mean_concretness"]
    df_clustering = df[selected_features]
    df_clustering = normalize(df_clustering, df_clustering.columns)
    for i in ["de", "en"]:
        print("Language", i)
        df_temp = df_clustering[df_clustering["language"]==i]
        df_temp = df_temp.drop(columns="language")

        labels = cluster(df_temp, method="KMeans")
        df_label = df[df["language"]==i]
        df_label["cluster"] = labels
        # df_groupby = df_label.groupby(by="author")["cluster"].apply(to_dict)
        df_groupby = df_label.groupby(by=["author", "cluster"])["cluster"].apply(sum)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df_groupby.dropna())
        df_groupby.to_csv(os.path.join(os.getcwd(), "data", "ML Results", "%s_cluster_Result.csv" % i))
            
        visualize_cluster(df_temp, labels, language=i)
    
def cluster(df, method="DBSCAN", n_cluster=5):
    """
    The clustering function, with which a set of methods can be applied to a dataset, currently DBSCAN and KMeans are implemented
    :param df: pandas dataframe of scores
    :param method: name of the cluster method
    :param n_cluster: number of clusters to search for in KMeans
    :return: the labels, to which cluster each element belongs
    """
    if method == "DBSCAN":
        model = DBSCAN(eps=0.01, min_samples=5)
        model.fit(df)
        return model.labels_
    elif method == "KMeans":
        model = KMeans(n_clusters=n_cluster, n_jobs=-1, n_init=100)
        model.fit(df)
        return model.labels_
    
def normalize(df, columns):
    """
    Normalizes data to a range 0-1, based on the data range given in the data
    :param df: pandas dataframe of scores
    :param columns: the columns which need normalization
    :return: pandas dataframe with normalized scores
    """
    for column in columns:
        try:
            df.loc[:,column] = df[column] / df[column].abs().max()
        except:
            pass
    return df

def max_cluster(list_of_elements):
    """
    Find the cluster with the most elements
    :param list_of_elements: list of labels
    :return: the cluster label and its count
    """
    count_dict ={}
    for i in list_of_elements:
        count_dict[i] = count_dict.get(i, 0) + 1
    count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    print(count_dict[0])
    return count_dict[0]

def visualize_cluster(df, label, language):
    """
    generate a 2D representation of the data using TSNE, coloring each cluster differently
    :param df: pandas dataframe of scores
    :param label: cluster labels for each entry
    :param language: the language this plot is generated for
    :return: None, saves a TSNE representation file and a figure
    """
    
    if os.path.isfile(os.path.join(os.getcwd(), "data", "ML Results", "%s_TSNE.csv" % language)):
        df_plotting = pd.read_csv(os.path.join(os.getcwd(), "data", "ML Results", "%s_TSNE.csv" %language))
        df_plotting["Label"] = label
    else:
        model = TSNE(n_components=2, n_jobs=-1)
        X_embedding = model.fit_transform(df)
        df_plotting= pd.DataFrame(data={"X": [i[0] for i in X_embedding],
                                        "Y": [i[1] for i in X_embedding], "Label": label})
        df_plotting.to_csv(os.path.join(os.getcwd(), "data", "ML Results", "%s_TSNE.csv" % language))
    # print(X_embedding)
    sns.scatterplot(x="X", y="Y", hue="Label", data=df_plotting)
    plt.title("Clustering for %s" % language)
    plt.savefig(os.path.join(os.getcwd(), "data", "ML Results", "%s_cluster_Result.png" % language), dpi=400)
    plt.show()

run_clustering()