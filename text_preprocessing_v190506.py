import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import matplotlib.pyplot as plt

stopwords = stopwords.words('french')
stopwords += ["les", "a", "donc", "ils"]
lemmatizer = FrenchLefffLemmatizer()

def lowercase(text):
    try:
        return ' '.join(word.lower() for word in text.split(' '))
    except:
        return np.nan

def remove_special_characters(text):
    try:
        text = re.sub(r"[^\w\s]", r" ", text)
        text = re.sub(r"\s+", r" ", text)
        text = re.sub(r"[0-9]+", r"", text)
        return text.strip()
    except:
        return np.nan

def remove_stopwords(text, stopwords):
    try:
        return ' '.join(word for word in text.split(' ') if word not in stopwords)
    except:
        return np.nan

def lemmatize_unigrams(text):
    try:
        return ' '.join(lemmatizer.lemmatize(word) for word in text.split(' '))
    except:
        return np.nan

def join_collocation(text, collocation):
    try:
        return re.sub(collocation, "_".join(collocation.split(" ")), text)
    except:
        return np.nan

def preprocess(text, stopwords=stopwords, collocations=[]):
    if str(text) == 'nan':
        return np.nan

    text = lowercase(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text, stopwords)
    text = lemmatize_unigrams(text)
    for collocation in collocations:
        text = join_collocation(text, collocation)
    text = remove_stopwords(text, stopwords)
    return text

def preprocess_texts_in_dataframe(df, columns, stopwords=stopwords, collocations=[]):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expecting a pandas DataFrame for df argument")
    if not isinstance(columns, list):
        raise ValueError("Expecting a list for columns argument")

    for column in columns:
        if column not in df.columns:
            raise ValueError(str(column) + " doesn't exist in dataframe's columns")
        if not isinstance(df[column], object):
            raise ValueError(str(column) + " column is not a text column")

        for index, text in df[column].iteritems():
            df.at[index, column] = preprocess(text, stopwords=stopwords, collocations=collocations)

    return df
