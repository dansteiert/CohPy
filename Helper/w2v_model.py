'''
Word2Vec:
It transforms the words into a vectorspace, by clustering "coocurring"/in the same context elements nearer together ->
the vector space can be limited to a number of dimensions (how many though)
One then can assume, that words with a closer relationship, will from cohesion within and over documents.
Some videos to watch: https://www.youtube.com/watch?v=D-ekE-Wlcds
https://www.youtube.com/watch?v=ERibwqs9p38

Distance evalution by "cosine similarity score"
'''
from gensim.models import KeyedVectors
from scipy import spatial
from Helper.Helper_functions import *
import os


def load_w2v(path_to_model):
    keyvector_path = path_to_model[:-4] + ".kv"
    if os.path.isfile(keyvector_path):
        w2v = KeyedVectors.load(keyvector_path, mmap="r")
        return w2v
    w2v = KeyedVectors.load_word2vec_format(path_to_model)
    w2v.save(fname_or_handle=path_to_model[:-4] + ".kv")
    return w2v


def sentence_similarity(sent_a_dict, sent_b_dict, sentiment_dict):
    vec_a_list = [v * sentiment_dict.get(k, 0) for k, v in sent_a_dict.items() if k in sentiment_dict]
    vec_b_list = [v * sentiment_dict.get(k, 0) for k, v in sent_b_dict.items() if k in sentiment_dict]
    vec_a = np.add.reduce(vec_a_list)
    vec_b = np.add.reduce(vec_b_list)

    similarity = spatial.distance.cosine(vec_a, vec_b)
    if similarity != similarity:
        return None
    return similarity


def sentiment_scores(frequency_dict_by_doc, w2v_model):
    if w2v_model is None:
        return None, None
    senti_dict = {k: w2v_model.get_vector(k) for k in frequency_dict_by_doc.keys() if k  in w2v_model.vocab}
    try:
        hitrate = len(senti_dict)/len(frequency_dict_by_doc)
    except:
        hitrate = None
    return senti_dict, hitrate