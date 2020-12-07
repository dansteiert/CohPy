from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, LsiModel, LdaModel
from gensim import corpora



def preprocessing(corpus_tokens):
    dictionary = corpora.Dictionary(corpus_tokens)
    doc_freq_matrix = [dictionary.doc2bow(doc) for doc in corpus_tokens]



    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, stop_words=None, lowercase=False, vocabulary=dictionary)
    tfidf = vectorizer.fit_transform(corpus_tokens)
    return (dictionary, doc_freq_matrix, tfidf)