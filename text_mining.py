import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def get_m_most_frequent_ngrams(text, ngram_range=(1,1), m=20, display=True):
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=m)
    X = np.array(vectorizer.fit_transform([text]).todense())[0]
    X = pd.DataFrame(X, index=vectorizer.get_feature_names(), columns=["frequency"])
    if display:
        X.loc["total words"] = len(text.split(' '))
        X = X.sort_values(by="frequency", ascending=False)
    else:
        X = X.sort_values(by="frequency", ascending=False)
        X = dict(X['frequency'])
    return X

def plot_wordcloud(text, ngram_range=(1,1), max_words=200, strict_stopwords=None):
    if strict_stopwords:
        text = ' '.join(word for word in text.split(' ') if word not in strict_stopwords)

    words_and_frequencies = get_m_most_frequent_ngrams(text, ngram_range=ngram_range, m=max_words, display=False)
    wordcloud = WordCloud(
        width=2000, height=1000, background_color=None,
        mode='RGBA', colormap='gnuplot', min_font_size=6
    ).generate_from_frequencies(words_and_frequencies)
    fig, ax = plt.subplots(figsize=(10,5))
    plt.imshow(wordcloud)
    plt.axis("off")

    return fig

def vectorize(corpus, dim=None, vectorization_type=1, display=False):
    if dim is None:
        vocab_size = len(pd.unique([word for text in corpus for word in text.split()]))
        if vocab_size > 150:
            dim = int(np.ceil(np.median([len(text.split(' ')) for text in corpus])))

    if vectorization_type == 1 or vectorization_type == 2:
        vectorizer = CountVectorizer(max_features=dim)
    elif vectorization_type == 3:
        vectorizer = TfidfVectorizer(max_features=dim, use_idf=False)
    elif vectorization_type == 4:
        vectorizer = TfidfVectorizer(max_features=dim)

    X = np.array(vectorizer.fit_transform(corpus).todense())
    if vectorization_type == 1:
        X[X != 0] = 1

    if display:
        X = pd.DataFrame(X, columns=vectorizer.get_feature_names())

    return X

def tokenize(corpus, pad=False, dim=None, with_dictionary=False, max_vocabulary=None):
    vectorizer = CountVectorizer(max_features=max_vocabulary)
    vectorizer.fit_transform(corpus)

    vocabulary = {value: i+1 for (i, value) in enumerate(vectorizer.get_feature_names())}
    default_index = np.max(list(vocabulary.values())) + 1
    corpus = [[vocabulary.get(word, default_index) for word in text.split(' ')] for text in corpus]

    if pad:
        max_length = np.max([len(text) for text in corpus])
        corpus = np.array([text + [0] * (max_length - len(text)) for text in corpus])
        if dim:
            corpus = corpus[:,:dim]

    if with_dictionary:
        return corpus, vocabulary
    else:
        return corpus
