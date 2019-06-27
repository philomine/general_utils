################################################################################
# The text_preprocessing module wraps a few functions useful to clean texts
# Packages you may need to install : french_lefff_lemmatizer, nltk
################################################################################

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
        text = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\\]\^_`{|}~\t\n]', r'', text)
        text = re.sub(r"[^\w\s]", r" ", text)
        text = remove_double_space(text)
        return text.strip()
    except:
        return np.nan

def remove_double_space(text):
    try:
        text = re.sub(r"\s+", r" ", text)
        return text.strip()
    except:
        return np.nan

def remove_numbers(text):
    try:
        text = re.sub(r"[0-9]+", r"", text)
        text = remove_double_space(text)
        return text.strip()
    except:
        return np.nan

def remove_stopwords(text, stopwords=stopwords):
    try:
        text = ' '.join(word for word in text.split(' ') if word not in stopwords)
        text = remove_double_space(text)
        return text
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

def preprocess_text(text, english=True, my_collocations=None, preprocessing_steps=[1, 2, 3, 4, 5, 6, 4], _settings=True):
    '''
    :param preprocessing_steps: Preprocessing steps to execute.
    1: lowercase,
    2: remove_special_characters,
    3: remove_numbers,
    4: remove_stopwords,
    5: lemmatize_unigrams,
    6: join_collocations
    '''
    if _settings:
        if my_collocations:
            global collocations
            collocations = my_collocations

        if not english:
            global stopwords
            stopwords = stopwords_fr
            global lemmatizer
            lemmatizer = lemmatizer_fr

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

def preprocess_corpus(corpus, english=True, my_collocations=None, preprocessing_steps=[1, 2, 3, 4, 5, 6, 4]):
    '''
    :param preprocessing_steps: Preprocessing steps to execute.
    1: lowercase,
    2: remove_special_characters,
    3: remove_numbers,
    4: remove_stopwords,
    5: lemmatize_unigrams,
    6: join_collocations
    '''
    if my_collocations:
        global collocations
        collocations = my_collocations

    if not english:
        global stopwords
        stopwords = stopwords_fr
        global lemmatizer
        lemmatizer = lemmatizer_fr

    res = []
    for text in corpus:
        res.append(preprocess_text(text, preprocessing_steps=preprocessing_steps, _settings=False))

    return res
