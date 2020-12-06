from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, LsiModel, LdaModel
from gensim import corpora



dictionary = corpora.Dictionary(tokens)
dt_matrix = [dictionary.doc2bow(doc) for doc in tokens]



vectorizer = TfidfVectorizer(tokenizer=lambda x: x, stop_words=None, lowercase=False, vocabulary=dictionary)
tfidf = vectorizer.fit_transform(tokens)