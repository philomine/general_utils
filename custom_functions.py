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

def progress_bar(iteration, total, length=50, empty_character=' ', fill_character='=', title='Progress'):
    '''
    Careful: iteration is meant to go from 0 to total-1, just like in the example
    :Example:
    >>> for i, item in enumerate(items):
    >>>     progress_bar(i, len(items))
    >>>     # Do stuff...
    Progress: [==================================================] 100.0%
    '''
    # Computing useful values
    percent = ("{0:5.1f}").format(100 * ((iteration+1) / float(total)))
    filled_length = int(length * (iteration+1) // total)
    bar = fill_character * (filled_length-1) + ">" + empty_character * (length - filled_length)

    # Updating printed output
    print('%s: [%s] %s%% ' % (title, bar, percent), end='\r')

    # Printing a new line at the end of the loop
    if iteration+1 == total:
        print()

    return None
