import pandas as pd
from Helper_functions import *

def load_score_file(path_to_file):
    df = pd.read_csv(path_to_file, sep=",")
    return df


def mean_concretness(lemma, list_dict):
    hitrate = 0
    conc = []
    for index, i in enumerate(lemma):
        temp = list_dict.get(i, None)
        # temp = concretness(lemma=i, df=df)
        if temp is not None:
            conc.append(float(temp.replace(",", ".")))
            hitrate += 1
    # print(conc)
    hitrate /= len(lemma)
    mean_concretness = mean_of_list(conc)
    return mean_concretness, hitrate

