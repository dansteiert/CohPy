'''
Word2Vec:
It transforms the words into a vectorspace, by clustering "coocurring"/in the same context elements nearer together ->
the vector space can be limited to a number of dimensions (how many though)
One then can assume, that words with a closer relationship, will from cohesion within and over documents.
Some videos to watch: https://www.youtube.com/watch?v=D-ekE-Wlcds
https://www.youtube.com/watch?v=ERibwqs9p38

Distance evalution by "cosine similarity score"
'''
from gensim.models import Word2Vec, KeyedVectors
import gensim
import numpy as np
from scipy import spatial
import multiprocessing
from Helper_functions import *


def train_word2vec(training_data):
    # build a large model on some large amount of german texts?
    # is their already a model
    cores = multiprocessing.cpu_count()
    w2v = Word2Vec(workers=cores - 1, window=5, min_count=2)
    w2v.build_vocab(sentences=training_data)
    w2v.train(training_data, total_examples=w2v.corpus_count, epochs=100)
    return w2v


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
        # print("to few elements:", len(sent_a_lemma), len(sent_b_lemma))
        # if len(sent_a_lemma) > 0 and len(sent_b_lemma) > 0:
            # print(sent_a_lemma)
            # print(sent_b_lemma)
        return 0, 0
    if similarity != similarity:
        if hit_a == 0 or hit_b == 0:
            # print("No hits found")
            return 0, 0
        # print("NAN result", similarity, vec_a, vec_b, sep="\n")
        return 0, 0
    hit_ratio = ((hit_a/searched_a) + (hit_b/searched_b))/2
    return similarity, hit_ratio


def sentence_vector_avg(w2v, lemma, tags,
                        accept_tags=[], accept_tags_start_with=[],
                        exclude_tags=[], exclude_tags_start_with=[]):
    hit = 0
    searched = 0
    sent_a_vec = np.zeros((w2v.vector_size,), dtype="float32")
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