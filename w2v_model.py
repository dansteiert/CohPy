'''
Word2Vec:
It transforms the words into a vectorspace, by clustering "coocurring"/in the same context elements nearer together ->
the vector space can be limited to a number of dimensions (how many though)
One then can assume, that words with a closer relationship, will from cohesion within and over documents.
Some videos to watch: https://www.youtube.com/watch?v=D-ekE-Wlcds
https://www.youtube.com/watch?v=ERibwqs9p38

Distance evalution by "cosine similarity score"
'''
from gensim.models import Word2Vec
import gensim
import multiprocessing


def train_word2vec(training_data):
    # build a large model on some large amount of german texts?
    # is their already a model
    cores = multiprocessing.cpu_count()
    w2v = Word2Vec(workers=cores-1, window=5, min_count=2)
    w2v.build_vocab(sentences=training_data)
    w2v.train(training_data, total_examples=w2v.corpus_count, epochs=100)
    return w2v

def load_w2v(path_to_model):
    w2v = gensim.models.Word2Vec.load(path_to_model)
    return w2v

def sentence_similarity(w2v, sent_a, sent_b):
    ## Greedy always take the closest related value?
    sent_sim = 0
    for i in sent_a:
        temp_max = 0
        for j in sent_b:
            sim = w2v.similarity(i, j)
            if sim > temp_max:
                temp_max = sim
        sent_sim += temp_max
    sent_sim /= len(sent_a)
    return sent_sim




