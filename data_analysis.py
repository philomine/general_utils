import string

import numpy as np
import pandas as pd


def string_format(val):
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


def dist(x, normalize=False):
    u, c = np.unique(x, return_counts=True)
    if normalize:
        total = np.sum(c)
        c = c / total
    return pd.DataFrame({"value": u, "freq": c})
