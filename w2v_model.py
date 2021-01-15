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
import multiprocessing
from Helper_functions import *


def train_word2vec(training_data):
    # build a large model on some large amount of german texts?
    # is their already a model
    cores = multiprocessing.cpu_count()
    w2v = Word2Vec(workers=cores-1, window=5, min_count=2)
    w2v.build_vocab(sentences=training_data)
    w2v.train(training_data, total_examples=w2v.corpus_count, epochs=100)
    return w2v

def load_w2v(path_to_model):
    w2v = KeyedVectors.load_word2vec_format(path_to_model)
    return w2v

def sentence_similarity(w2v, sent_a_lemma, sent_a_tags, sent_b_lemma, sent_b_tags,
                        accept_tags=[], accept_tags_start_with=[],
                        exclude_tags=[], exclude_tags_start_with=[]):
    ## Greedy always take the closest related value?
    hit = 0
    searched = 0
    sent_sim = []
    for l_a, t_a in zip(sent_a_lemma, sent_a_tags):
        if check_tags(tag=t_a, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                      exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with):
            if l_a not in w2v.vocab:
                searched += 1
                continue
            temp_max = 0
            elem = -1
            for index, (l_b, t_b) in enumerate(zip(sent_b_lemma, sent_b_tags)):
                if check_tags(tag=t_b, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                              exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with):
                    try:
                        sim = w2v.similarity(l_a, l_b)
                        if sim > temp_max:
                            temp_max = sim
                            elem = index
                            searched += 1
                            hit += 1
                    except:
                        searched += 1
                        continue
            if temp_max > 0:
                sent_b_lemma.pop(elem)
                sent_sim.append(temp_max)
            if len(sent_b_lemma) == 0:
                break

    return mean_of_list(sent_sim), hit, searched




