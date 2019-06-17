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
    ax.imshow(wordcloud)
    ax.axis("off")

    return fig

def vectorize(corpus, dim=None, vectorization_type=1, display=False):
    """
    :Example:
    >>> corpus = [
    >>>     'this is the first document',
    >>>     'this document is the second document',
    >>>     'here is the third one',
    >>>     'is this the first document'
    >>> ]
    >>>
    >>> # Default behavior
    >>> vectorize(corpus)
    [[1 1 0 1 0 0 1 0 1]
     [1 0 0 1 0 1 1 0 1]
     [0 0 1 1 1 0 1 1 0]
     [1 1 0 1 0 0 1 0 1]]
    >>>
    >>> # The display option returns a pandas DataFrame
    >>> vectorize(corpus, display=True)
       document  first  here  is  one  second  the  third  this
    0         1      1     0   1    0       0    1      0     1
    1         1      0     0   1    0       1    1      0     1
    2         0      0     1   1    1       0    1      1     0
    3         1      1     0   1    0       0    1      0     1
    >>>
    >>> # The vectorization_type parameter determines the encoding type, defaults to 1
    >>> #   1: One hot encoding
    >>> #   2: Counting words
    >>> #   3: Word frequency
    >>> #   4: Term frequency inverse document frequency
    >>> vectorize(corpus, display=True, vectorization_type=2)
       document  first  here  is  one  second  the  third  this
    0         1      1     0   1    0       0    1      0     1
    1         2      0     0   1    0       1    1      0     1
    2         0      0     1   1    1       0    1      1     0
    3         1      1     0   1    0       0    1      0     1
    >>>
    >>> # The dim parameter determines the number of dimensions, taking the dim most used words
    >>> vectorize(corpus, display=True, vectorization_type=3, dim=4)
       document    is   the  this
    0      0.25  0.25  0.25  0.25
    1      0.40  0.20  0.20  0.20
    2      0.00  0.50  0.50  0.00
    3      0.25  0.25  0.25  0.25
    """
    if dim is None:
        vocab_size = len(pd.unique([word for text in corpus for word in text.split()]))
        if vocab_size > 150:
            dim = int(np.ceil(np.median([len(text.split(' ')) for text in corpus])))

    if vectorization_type == 1 or vectorization_type == 2:
        vectorizer = CountVectorizer(max_features=dim)
    elif vectorization_type == 3:
        vectorizer = TfidfVectorizer(max_features=dim, use_idf=False, norm='l1')
    elif vectorization_type == 4:
        vectorizer = TfidfVectorizer(max_features=dim, norm='l1')

    X = np.array(vectorizer.fit_transform(corpus).todense())
    if vectorization_type == 1:
        X[X != 0] = 1

    if display:
        X = pd.DataFrame(X, columns=vectorizer.get_feature_names())

    return X

def tokenize(corpus, pad=False, dim=None, with_dictionary=False, max_vocabulary=None):
    """
    If pad is False, dim is ignored

    :Example:
    >>> corpus = [
    >>>     'this is the first document',
    >>>     'this document is the second document',
    >>>     'here is the third one',
    >>>     'is this the first document'
    >>> ]
    >>>
    >>> # Default behavior
    >>> tokenize(corpus)
    [[9, 4, 7, 2, 1], [9, 1, 4, 7, 6, 1], [3, 4, 7, 8, 5], [4, 9, 7, 2, 1]]
    >>>
    >>> # The with_dictionary parameter
    >>> tokenize(corpus, with_dictionary=True)
    ([[9, 4, 7, 2, 1], [9, 1, 4, 7, 6, 1], [3, 4, 7, 8, 5], [4, 9, 7, 2, 1]],
     {'document': 1,
      'first': 2,
      'here': 3,
      'is': 4,
      'one': 5,
      'second': 6,
      'the': 7,
      'third': 8,
      'this': 9})
    >>> tokenized_corpus, vocabulary = tokenize(corpus, with_dictionary=True)
    >>>
    >>> # The pad and dim parameters
    >>> tokenize(corpus, pad=True)
    [[9, 4, 7, 2, 1, 0],
     [9, 1, 4, 7, 6, 1],
     [3, 4, 7, 8, 5, 0],
     [4, 9, 7, 2, 1, 0]]
    >>> tokenize(corpus, pad=True, dim=4)
    [[9, 4, 7, 2],
     [9, 1, 4, 7],
     [3, 4, 7, 8],
     [4, 9, 7, 2]]
    >>>
    >>> # The max_vocabulary parameter
    >>> tokenize(corpus, max_vocabulary=3)
    [[4, 2, 3, 4, 1], [4, 1, 2, 3, 4, 1], [4, 2, 3, 4, 4], [2, 4, 3, 4, 1]]
    >>> tokenize(corpus, max_vocabulary=3, pad=True)
    [[4, 2, 3, 4, 1, 0],
     [4, 1, 2, 3, 4, 1],
     [4, 2, 3, 4, 4, 0],
     [2, 4, 3, 4, 1, 0]]
    >>> tokenize(corpus, max_vocabulary=3, dim=4, pad=True)
    [[4, 2, 3, 4],
     [4, 1, 2, 3],
     [4, 2, 3, 4],
     [2, 4, 3, 4]]
    """
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
