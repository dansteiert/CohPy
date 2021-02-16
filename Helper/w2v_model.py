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



def load_w2v(path_to_model):
    w2v = KeyedVectors.load_word2vec_format(path_to_model)
    return w2v


def sentence_similarity(w2v, sent_a_lemma, sent_a_tags, sent_b_lemma, sent_b_tags,
                        accept_tags=[], accept_tags_start_with=[],
                        exclude_tags=[], exclude_tags_start_with=[]):
    vec_a, hit_a, searched_a = sentence_vector_avg(w2v=w2v, lemma=sent_a_lemma, tags=sent_a_tags,
                        accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                        exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with)
    vec_b, hit_b, searched_b = sentence_vector_avg(w2v=w2v, lemma=sent_b_lemma, tags=sent_b_tags,
                        accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                        exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with)
    similarity = spatial.distance.cosine(vec_a, vec_b)
    if searched_a == 0 or searched_b == 0:

        return 0, 0
    if similarity != similarity:
        if hit_a == 0 or hit_b == 0:
            return 0, 0
        return 0, 0
    hit_ratio = ((hit_a/searched_a) + (hit_b/searched_b))/2
    return similarity, hit_ratio


def sentence_vector_avg(w2v, lemma, tags,
                        accept_tags=[], accept_tags_start_with=[],
                        exclude_tags=[], exclude_tags_start_with=[]):
    hit = 0
    searched = 0
    try:
        sent_a_vec = np.zeros((w2v.vector_size,), dtype="float32")
    except:
        print("No w2v model given!")
        return 0, 0, 0
    for l_a, t_a in zip(lemma, tags):
        if check_tags(tag=t_a, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                      exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with):
            if l_a not in w2v.vocab:
                searched += 1
                continue
            searched += 1
            hit += 1
            sent_a_vec = np.add(sent_a_vec, w2v.get_vector(l_a))
            if hit > 0:
                sent_a_vec = np.divide(sent_a_vec, hit)
    return sent_a_vec, hit, searched