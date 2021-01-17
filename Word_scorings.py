import pandas as pd


def load_score_file(path_to_file):
    df = pd.read_csv(path_to_file, sep=",")
    return df


def Concretness(lemma, df):
    pos = find_lemma(lemma=lemma, search_list=df["Word"].tolist())
    if pos is None:
        return None
    value = df.loc[pos, "AbstConc"]
    return float(value.replace(",", "."))


def find_lemma(lemma, search_list, pos=0):
    lemma = lemma.lower()
    if len(search_list) == 0:
        return None
    search_list_mid = len(search_list) // 2

    mid_word = search_list[search_list_mid].lower()
    if mid_word == lemma:
        pos += search_list_mid
        return pos
    elif mid_word > lemma:
        return (find_lemma(lemma=lemma, search_list=search_list[:search_list_mid], pos=pos))
    elif mid_word < lemma:
        return (find_lemma(lemma=lemma, search_list=search_list[search_list_mid + 1:], pos=pos + search_list_mid + 1))
