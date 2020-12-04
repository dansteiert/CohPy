import nltk
from nltk.tokenize import TweetTokenizer
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import seaborn as sns
'''
Workflow:
- Tokenize the corpus
- Pos Tag the words
   o Potentially need to build the POS Tagger from scratch (train a model) -> need of labled data
- Stopwords removal (By POS tag or by a list of predefined words (nltk.corpus.stopwords("german")?)
- Lemmatization  (German?)
'''


def expand_contractions(text, contraction_map, token_type="sentence"):
    '''expand contractions isn't -> is not'''
    # contractions_pattern = re.compile('({})'.format('|'.join(contraction_map.keys())), flags=re.IGNORECASE|re.DOTALL)
    tokenizer = TweetTokenizer()
    if token_type == "corpus":

        sentences = nltk.sent_tokenize(text)
        word_tokens = [tokenizer.tokenize(sentence) for sentence in sentences]
        new_tokens = []
        for sentence in word_tokens:
            word_list = []
            for word in sentence:
                contraction = contraction_map.get(word, None)
                if contraction is not None:
                    word_list.extend(tokenizer.tokenize(contraction))
                else:
                    word_list.append(word)
            new_tokens.append(word_list)
        expanded_sentence = []

    elif token_type == "sentence":
        word_tokens = [tokenizer.tokenize(sentence) for sentence in text]
        new_tokens = []
        for word in word_tokens:
            contraction = contraction_map.get(word, None)
            if contraction is not None:
                new_tokens.extend(tokenizer.tokenize(contraction))
            else:
                new_tokens.append(word)
    else:
        print("token_type must be either corpus or sentence")
        new_tokens = None
    return new_tokens


def tokenize_text(text, token_type="sentence"):
    '''splitting the text into sentences and the sentences into words'''
    # create sentences
    if token_type == "corpus":
        sentences = nltk.sent_tokenize(text)
        # tokenize words
        word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    elif token_type == "sentence":
        word_tokens = [nltk.word_tokenize(sentence) for sentence in text]
    else:
        print("token_type must be either corpus or sentence")
        word_tokens = None
    return word_tokens


def remove_characters_after_tokenization(tokens, token_type="sentence", pos_tagged=False):
    '''filter out punctuation - !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    if not pos_tagged:

        if token_type == "sentence":
            filtered_tokens = list(filter(None, [pattern.sub('', token) for token in tokens]))
        elif token_type == "corpus":
            filtered_tokens = []
            for i in tokens:
                filtered_tokens.append(list(filter(None, [pattern.sub('', token) for token in i])))
        else:
            print("token_type must be either corpus or sentence")
            filtered_tokens = None
    else:
        if token_type == "sentence":
            filtered_tokens = list(filter(None, [(pattern.sub('', token[0]), token[1]) for token in tokens]))
        elif token_type == "corpus":
            filtered_tokens = []
            for i in tokens:
                filtered_tokens.append(list(filter(None, [(pattern.sub('', token[0]), token[1]) for token in i])))
        else:
            print("token_type must be either corpus or sentence")
            filtered_tokens = None

    return filtered_tokens


def remove_stopwords(tokens, token_type="sentence", pos_tagged=False, stopword_list=None, pos_tags_to_keep=None):
    '''
    Either remove stopwords (given ones or from nltk.corpus.stopwords.words("english")) or keep only pos_tags in pos_Tags_to_keep
    remove stop words like:
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
     'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
     "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
     'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
      'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
       'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
         'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
          'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
           't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
            'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
             'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
              'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
               'won', "won't", 'wouldn', "wouldn't"]

    '''
    if stopword_list is None:
        stopword_list = nltk.corpus.stopwords.words("english")


    if not pos_tagged:
        if token_type == "sentence":
            filtered_tokens = [i for i in tokens if i not in stopword_list]
        elif token_type == "corpus":
            filtered_tokens = []
            for i in tokens:
                filtered_tokens.append([token for token in i if token not in stopword_list])
        else:
            print("token_type must be either corpus or sentence")
            filtered_tokens = None
        return filtered_tokens
    else:
        if pos_tags_to_keep is not None:
            if token_type == "sentence":
                filtered_tokens = [token for token in tokens if token[1] in pos_tags_to_keep]
            elif token_type == "corpus":
                filtered_tokens = []
                for i in tokens:
                    filtered_tokens.append([token for token in i if token[1] in pos_tags_to_keep])
            else:
                print("token_type must be either corpus or sentence")
                filtered_tokens = None
        else:
            if token_type == "sentence":
                filtered_tokens = [token for token in tokens if token[0] not in stopword_list]
            elif token_type == "corpus":
                filtered_tokens = []
                for i in tokens:
                    filtered_tokens.append([token for token in i if token[0] not in stopword_list])
            else:
                print("token_type must be either corpus or sentence")
                filtered_tokens = None
        return filtered_tokens


def lemmatization(tokens, token_type="sentence"):
    ''' Cast to wordnet pos tags and lemmatize it with the WordNetLemmatizer'''

    post_tag_dict = {"NOUN": wn.NOUN, "ADV": wn.ADV, "ADJ": wn.ADJ, "VERB": wn.VERB, "PRON": wn.NOUN}
    lemmatizer = WordNetLemmatizer()
    if token_type == "sentence":
        lemmatized_tokens = [lemmatizer.lemmatize(word=token[0], pos=post_tag_dict.get(token[1], wn.NOUN)) for token in tokens]
    elif token_type == "corpus":
        lemmatized_tokens = []
        for i in tokens:
            lemmatized_tokens.append([lemmatizer.lemmatize(word=token[0], pos=post_tag_dict.get(token[1], wn.NOUN)) for token in i])
    else:
        print("token_type must be either corpus or sentence")
        lemmatized_tokens = None
    return lemmatized_tokens


def tfidf(tokens):

    vocabulary = list(set([i for j in tokens for i in j]))

    tf, idf, tfidf_values = own_tfidf_implementation(tokens=tokens, vocabulary=vocabulary)
    print(tf)
    print(idf)
    print(tfidf_values)

    return tf, idf, tfidf_values, vocabulary


def own_tfidf_implementation(tokens, vocabulary):
    tf = np.zeros(shape=(len(tokens), len(vocabulary)))
    df = np.zeros(len(vocabulary))
    for document, token_list in enumerate(tokens):
        for voc, word in enumerate(vocabulary):
            if word in token_list:
                tf[document][voc] += 1
                df[voc] += 1
        tf[document] = tf[document]/len(token_list)
    idf = [1 + np.log(len(tokens)/ (1 + i)) for i in df]
    tfidf_values = tf.dot(idf)
    return tf, idf, tfidf_values


def corpus_to_cluster(tokens, vocabulary, tf, idf, tfidf_values, X_embedded):
    tf[tf == np.inf] = 0
    tf[np.isnan(tf)] = 0

    tf_matrix = tf.astype(dtype=np.float64)
    dbscan = DBSCAN(eps=2, min_samples=3)
    dbscan.fit(X=tf_matrix)
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X=tf_matrix)


    df = pd.DataFrame(data={"DBSCAN": dbscan.labels_, "KMEANS": kmeans.labels_, "x": X_embedded[:, 0], "y": X_embedded[:, 1]})
    sns.set(rc={'figure.figsize': (15, 15)})



    # plot
    sns.scatterplot(data=df, x="x", y="y", hue="DBSCAN", palette=sns.color_palette("bright", len(list(set(dbscan.labels_)))))
    plt.title('t-SNE with no Labels')
    plt.savefig("t-sne_covid19.png")
    plt.show()
    plt.close()
    # plot
    sns.scatterplot(data=df, x="x", y="y", hue="KMEANS", palette=sns.color_palette("bright", len(list(set(kmeans.labels_)))))
    plt.title('t-SNE with no Labels')
    plt.savefig("t-sne_covid19.png")
    plt.show()
    plt.close()
    return df

def get_two_dimensional_representation(tf):
    tf[tf == np.inf] = 0
    tf[np.isnan(tf)] = 0

    tf_matrix = tf.astype(dtype=np.float64)
    tsne = TSNE(verbose=1, perplexity=100, random_state=42)
    X_embedded = tsne.fit_transform(tf_matrix)
    return X_embedded


