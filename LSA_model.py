'''
LSA:
It uses a term frequncy matrix and applys a Sigular value decomposition to find the top "x" columns within this matrix.
Each Row now represents a topic, which is quite similar to the LDA model, at least in its result.
'''

from gensim.models import LsiModel



def LSA(df_matrix, dictionary):
    '''
    Build the LDA model
    :param df_matrix: document term matrix
    :param dictionary: dictionary
    :return: the build model
    '''
    return LsiModel(corpus=df_matrix, num_topics=10, id2word=dictionary)