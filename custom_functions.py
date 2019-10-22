# import sys
# import os
# sys.path.append(os.path.expanduser("~/Projects/Python/python_custom_functions"))

import numpy as np
import pandas as pd
from scipy import stats

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

def progress_bar(i, n):
    '''
    Careful: i is meant to go from 0 to n-1, just like in the example
    :Example:
    >>> for i, item in enumerate(items):
    >>>     progress_bar(i, len(items))
    >>>     # Do stuff...
    [=================================================>] 100.0%
    '''
    if np.floor(1000 * (i+1)/n) != np.floor(1000 * i/n):
        advancement = (i+1) / n
        percent = "{0:5.1f}".format(100 * advancement)

        bar_size = 50
        filled_bar_size = int(np.ceil(advancement * bar_size))
        bar = "[" + (filled_bar_size - 1) * "=" + ">" + (bar_size - filled_bar_size) * " " + "]"

        print(bar + " " + percent + "%", end='\r')

    if i+1 == n:
        print("")

    return None

# Medium article "Dynamically add a method to a class" by Michael Garod
# https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
def add_method(cls):
    '''
    Add a method to a class.

    :Example: 
    # Adding method foo() to class 
    @add_method(A)
    def foo():
        print('hello world!')
    # Method foo() can still be used on its own (doesn't need self attribute).
    '''
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator
