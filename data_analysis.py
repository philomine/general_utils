# import sys
# import os
# sys.path.append(os.path.expanduser("~/Projects/Python/python_custom_functions"))

import numpy as np
import pandas as pd
import time
import datetime
import string
from scipy import stats
from functools import wraps # This convenience func preserves name and docstring

import plotly
import plotly.express as px
import plotly.graph_objects as go

def number_of_nan(X, axis = 0):
    X = np.array(X, dtype = 'float')
    return np.sum(np.isnan(X), axis=axis)

def clear_nans(X, dont_drop=None):
    number_of_nans_in_columns = number_of_nan(X, axis=0)
    if not len(np.unique(number_of_nans_in_columns)) == 1:
        outliers = np.abs(stats.zscore(number_of_nans_in_columns)) > 1
        columns_to_drop = X.columns[outliers]
        if dont_drop:
            columns_to_drop = [column for column in columns_to_drop if column not in dont_drop]
            X = X.drop(columns_to_drop, axis=1)

    X = X.dropna()

    return X

def distribution(values, title='Distribution of values', log_scale=False):
    unique, counts = np.unique([str(val) for val in values], return_counts=True)
    distribution = pd.DataFrame({'Count': counts, 'Values': unique})

    fig = go.Figure(data=[go.Bar(x=distribution['Values'], y=distribution['Count'], text=distribution['Count'], textposition='auto')])
    if log_scale:
        fig.update_layout(title=title, xaxis=dict(title='Values'), yaxis=dict(title='Count'), yaxis_type="log")
    else:
        fig.update_layout(title=title, xaxis=dict(title='Values'), yaxis=dict(title='Count'))
    fig.show()

def string_format(val):
    if not isinstance(val, str):
        res = np.nan
    else:
        res = ''
        for c in val:
            if c in string.digits:
                res += 'D'
            elif c in string.ascii_letters:
                res += 'L'
            elif c in string.punctuation:
                res += 'P'
            elif c in string.whitespace:
                res += 'S'
            else:
                res += 'O'
    return res

def dist(x, normalize=False):
    u, c = np.unique(x, return_counts=True)
    if normalize:
        total = np.sum(c)
        c = c / total
    return pd.DataFrame({'value': u, 'freq': c})