import string

import numpy as np
import pandas as pd


def get_string_format(val):
    """ Takes val and returns string format """
    if not isinstance(val, str):
        res = np.nan
    else:
        res = ""
        for c in val:
            if c in string.digits:
                res += "D"
            elif c in string.ascii_letters:
                res += "L"
            elif c in string.punctuation:
                res += "P"
            elif c in string.whitespace:
                res += "S"
            else:
                res += "O"
    return res


def get_dist_table(sample, normalize=False):
    """ Returns the dist table of a sample
    A dist table is a pd.DataFrame with two columns: 
        - value: list of values
        - freq: frequency of those values
    """
    u, c = np.unique(sample, return_counts=True)
    if normalize:
        total = np.sum(c)
        c = c / total
    return pd.DataFrame({"value": u, "freq": c})


def get_sample(dist_table):
    """ Returns the sample of a dist table 
    A dist table is a pd.DataFrame with two columns: 
        - value: list of values
        - freq: frequency of those values
    """
    sample = []
    for __, row in dist_table.iterrows():
        sample += [row["value"]] * int(row["freq"])
    return sample
