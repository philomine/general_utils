import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import matplotlib.pyplot as plt

stopwords_en = stopwords.words('english')
stopwords_fr = stopwords.words('french')
stopwords_fr += ["les", "a", "donc", "ils"]
lemmatizer_en = WordNetLemmatizer()
lemmatizer_fr = FrenchLefffLemmatizer()

stopwords = stopwords_en
lemmatizer = lemmatizer_en
collocations = []

def lowercase(text):
    try:
        return ' '.join(word.lower() for word in text.split(' '))
    except:
        return np.nan

def remove_special_characters(text):
    try:
        text = re.sub(r"<br />", r" ", text)
        text = re.sub(r"[^\w\s]", r" ", text)
        text = re.sub(r"\s+", r" ", text)
        return text.strip()
    except:
        return np.nan

def remove_numbers(text):
    try:
        text = re.sub(r"[0-9]+", r"", text)
        return text.strip()
    except:
        return np.nan

def remove_stopwords(text, stopwords=stopwords):
    try:
        return ' '.join(word for word in text.split(' ') if word not in stopwords)
    except:
        return np.nan

def lemmatize_unigrams(text):
    try:
        return ' '.join(lemmatizer.lemmatize(word) for word in text.split(' '))
    except:
        return np.nan

def _join_collocation(text, collocation):
    try:
        return re.sub(collocation, "_".join(collocation.split(" ")), text)
    except:
        return np.nan

def join_collocations(text, my_collocations=None):
    if my_collocations:
        global collocations
        collocations = my_collocations

    for collocation in collocations:
        text = _join_collocation(text, collocation)
    return text

def preprocess(text, my_collocations=None, preprocessing_steps=[1, 2, 3, 4, 5, 6, 4]):
    if my_collocations:
        global collocations
        collocations = my_collocations

    if str(text) == 'nan':
        return np.nan

    steps_dictionary = {
        1: lowercase,
        2: remove_special_characters,
        3: remove_numbers,
        4: remove_stopwords,
        5: lemmatize_unigrams,
        6: join_collocations
    }

    for step in preprocessing_steps:
        text = steps_dictionary[step](text)

    return text

def preprocess_texts_in_dataframe(df, columns, english=True, my_collocations=None, preprocessing_steps=[1, 2, 3, 4, 5, 6, 4]):
    df = df.copy()

    if my_collocations:
        global collocations
        collocations = my_collocations

    if not english:
        global stopwords
        stopwords = stopwords_fr
        global lemmatizer
        lemmatizer = lemmatizer_fr

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
            df.at[index, column] = preprocess(text, preprocessing_steps=preprocessing_steps)

    return df
