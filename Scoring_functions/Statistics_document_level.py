from Helper.Helper_functions import search_tag_set, to_count_dict, mean_of_list
import numpy as np




def logical_incidence(tagsets_by_doc, tagset_name, doc_words):
    """
    Ref: Grasser2004 - Logical Operators
    Calculate the incidence of logicals, within the document
    :param tagsets_by_doc: dict, {key=tagset names, value=dict{key=lemma, value=absolute word count}}
    :param tagset_name: str, tagset name for the logicals
    :param doc_words: int, # of words within the document
    :return: dict, {key=name of logical, value=incidence of this logical}
    """
    tagset = tagsets_by_doc.get(tagset_name, {})
    
    normalizer = doc_words / 1000
    # tagset["all"] = sum(tagset.values())
    return {"Incidence logicals": sum(tagset.values())/normalizer}


def connective_incidence(lemma, df_connective, connective_type_label):
    """
    Ref: Grasser2004 - Connectives
    Ref: Crossley2016- Connectives
    Calculate connective incidence scores for each connective type given in the dependency dataset
    :param lemma: list, [lemma of the document]
    :param df_connective: dict, {key=lemma, value=str(Type of Connective)}
    :param connective_type_label: str, label of the Type of Connective column
    :return: dict, {key=incidence connetive *Type of Connective*, value=Incidence of the connective type}
    """
    
    
    agg_list = []
    for index, l in enumerate(lemma):
        temp = []
        for i in range(0, 3):
            search_string = " ".join(lemma[index: index + i])
            temp_dict = df_connective.get(search_string, None)
            if temp_dict is not None:
                temp.append(temp_dict.get(connective_type_label, None))
        for i in reversed(temp):
            if i is not None:
                agg_list.append(i)
                break
    count_dict = to_count_dict(agg_list)
    all_conncetives = list(set([i.get("Connective Type", None) for i in df_connective.values()]))
    missing_connectives = [i for i in all_conncetives if i not in count_dict.keys()]
    count_dict = {**count_dict, **{i: 0 for i in missing_connectives}}
    normalizer = len(lemma) / 1000
    connective_incidences = {"incidence connective " + k: v/normalizer for k, v in count_dict.items()}
    
    return connective_incidences


def unique_lemma(tagsets_by_doc, tagset_name, document_words):
    """
    Ref: Crossley2016 - Givenness
    Calculate the unique lemma, within the document, can be reduced to a certain tagset
    :param tagset_by_doc: dict, {key=tagset names, value=dict{key=lemma, value=absolute word count}}
    :param tagset_name: str, tagset name, defining the POS tag set
    :param document_sentences: # of sentences
    :return: float, # uniquely occurring lemma/# sentences
    """
    tagset = tagsets_by_doc.get(tagset_name, {})
    list_unique = [True for v in tagset.values() if v == 1]
    return len(list_unique)/(document_words/1000)
    

def Flescher_Reading_Ease(document_words, document_syllables, num_sentences):
    """
    Ref: Grasser2004 - Readability Scores
    :param document_words: list, [str= words]
    :param document_syllables: list, [int=Syllabel count]
    :param num_sentences: int, # sentences in the document
    :return: float, FRE score: 206.835 - 1.015 * AvgSentenceLength - 84.6 * AvgSyllablePerWord
    """
    if len(document_words) < 200:
        return None

    ASL = len(document_words) / num_sentences  # ratio #words/#sent
    ASW = sum(document_syllables) / len(document_words)  # # ratio Syllables/Words
    return 206.835 - 1.015 * ASL - 84.6 * ASW


def Flescher_Kincaid_Grade_Level(document_words, document_syllables, num_sentences):
    """
    Ref: Grasser2004 - Readability Scores

    :param document_words: list, [str= words]
    :param document_syllables: list, [int=Syllabel count]
    :param num_sentences: int, # sentences in the document
    :return: float, FKGL score: 0.39 * AvgSentenceLength + 11.8 * AvgSyllablePerWord - 15.59
    """
    if len(document_words) < 200:
        return None

    ASL = len(document_words) / num_sentences  # ratio #words/#sent
    ASW = sum(document_syllables) / len(document_words)
    return 0.39 * ASL + 11.8 * ASW - 15.59




