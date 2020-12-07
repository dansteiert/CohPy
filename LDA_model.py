'''
What is LDA:
It assumes a list of topics, from which documents are generated. Find the most likely topics for each document.
Not sure yet how this will help with Cohesion!

Assumption: related concepts have similar topic probabilities based on underlying cooccurrence patterns.
What does this imply?
That the more words from a given topic a drawn, the larger the cohesion is?
'''

from gensim.models import LdaModel


def LDA(df_matrix, dictionary):
    '''
    Build the LDA model
    :param df_matrix: document term matrix
    :param dictionary: dictionary
    :return: the build model
    '''
    return LdaModel(corpus=df_matrix, id2word=dictionary, num_topics=10, iterations=100)