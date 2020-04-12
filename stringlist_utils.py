import numpy as np
import pandas as pd

from .data_analysis import dist


def stringlist_length(stringlist):
    if isinstance(stringlist, str) and (stringlist != ""):
        length = len(stringlist.split(";"))
    else:
        length = 0
    return length


def append_stringlists(stringlists):
    res = []
    for stringlist in stringlists:
        if isinstance(stringlist, str) and (stringlist != ""):
            res += stringlist.split(";")
    return res


def stringlists_dist(stringlists, normalize=True):
    values = np.unique(append_stringlists(stringlists))
    res = pd.Series(index=values, data=0)
    for stringlist in stringlists:
        if isinstance(stringlist, str) and (stringlist != ""):
            stringlist_dist = dist(stringlist.split(";"), normalize=normalize)
            stringlist_dist = stringlist_dist.set_index("value").freq
            res[stringlist_dist.index] += stringlist_dist.values
    return pd.DataFrame({"value": res.index, "freq": res.values})
