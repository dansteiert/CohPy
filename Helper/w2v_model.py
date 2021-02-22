from gensim.models import KeyedVectors
from scipy import spatial
from Helper.Helper_functions import *
import os


def load_w2v(path_to_model):
    '''
    Load a vector space model, if it is in the word2vec_format (.vec), write it out as a Keyed Vector (.kv)
    :param path_to_model: path to word2vec_format file
    :return: w2v model
    '''
    keyvector_path = path_to_model[:-4] + ".kv"
    if os.path.isfile(keyvector_path):
        w2v = KeyedVectors.load(keyvector_path, mmap="r")
        return w2v
    w2v = KeyedVectors.load_word2vec_format(path_to_model)
    w2v.save(fname_or_handle=path_to_model[:-4] + ".kv")
    return w2v


def sentence_sentiment_shift(sent_a_dict, sent_b_dict, sentiment_dict):
    '''
    Calculate the cosine distance between two vectors, or rather a list of two vectors
    :param sent_a_dict: dict, {key=lemma, value=occurance in sentence} 
    :param sent_b_dict: dict, {key=lemma, value=occurance in sentence}
    :param sentiment_dict: dict, {key=lemma, value=sentiment vector}
    :return: cosine distnace of two vectors, called the sentiment shift
    '''
    vec_a_list = [v * sentiment_dict.get(k, 0) for k, v in sent_a_dict.items() if k in sentiment_dict]
    vec_b_list = [v * sentiment_dict.get(k, 0) for k, v in sent_b_dict.items() if k in sentiment_dict]
    vec_a = np.add.reduce(vec_a_list)
    vec_b = np.add.reduce(vec_b_list)

    shift = spatial.distance.cosine(vec_a, vec_b)
    if shift != shift:
        return None
    return shift


def sentiment_scores(frequency_dict_by_doc, w2v_model):
    '''
    Preprocessing: Create a dictionary of all lemma in the document, and their sentiment vector from the w2v model
    :param frequency_dict_by_doc: dict {key=lemma, value=absolute count of occurance in the document}
    :param w2v_model: w2v model with sentiment vectors for each word
    :return: dict{key=lemma, value=sentiment vector}; float percentage of words which are in the w2v model
    '''
    if w2v_model is None:
        return None, None
    senti_dict = {k: w2v_model.get_vector(k) for k in frequency_dict_by_doc.keys() if k  in w2v_model.vocab}
    try:
        hitrate = len(senti_dict)/len(frequency_dict_by_doc)
    except:
        hitrate = None
    return senti_dict, hitrate