import numpy as np
import pandas as pd

from .data_analysis import get_dist_table


def stringlist_length(stringlist):
    """ Gives the length of a stringlist 
    A stringlist is a list of values concatenated in a string with ';' 
    separator. It's useful for labeling observations (an observation can have 
    0, 1 or several labels, which will be concatenated in a stringlist).

    Parameters
    ----------
    stringlist: str
        List of values concatenated in a string with ';' separator.
    """
    if isinstance(stringlist, str) and (stringlist != ""):
        length = len(stringlist.split(";"))
    else:
        length = 0
    return length


def stringlist_unique(stringlist):
    """ Applies np.unique to the stringlist 
    A stringlist is a list of values concatenated in a string with ';' 
    separator. It's useful for labeling observations (an observation can have 
    0, 1 or several labels, which will be concatenated in a stringlist).

    Parameters
    ----------
    stringlist: str
        List of values concatenated in a string with ';' separator.
    """
    if isinstance(stringlist, str) and (stringlist != ""):
        stringlist = ";".join(np.unique(stringlist.split(";")))
    else:
        stringlist = np.nan
    return stringlist


def append_stringlists(stringlists):
    """ Returns a stringlist composed of all stringlists concatenated
    A stringlist is a list of values concatenated in a string with ';' 
    separator. It's useful for labeling observations (an observation can have 
    0, 1 or several labels, which will be concatenated in a stringlist).

    Parameters
    ----------
    stringlist: array like of str
        List of values concatenated in a string with ';' separator.
    """
    res = []
    for stringlist in stringlists:
        if isinstance(stringlist, str) and (stringlist != ""):
            res += stringlist.split(";")
    if len(res) > 0:
        res = ";".join(res)
    else:
        res = np.nan
    return res


def stringlists_dist(stringlists, weighted=True, normalize=False):
    """ Returns the dist table of the stringlists
    A stringlist is a list of values concatenated in a string with ';' 
    separator. It's useful for labeling observations (an observation can have 
    0, 1 or several labels, which will be concatenated in a stringlist).

    Parameters
    ----------
    stringlist: array like of str
        List of values concatenated in a string with ';' separator.
    weighted: bool (optional, default: True)
        Should the final distribution be weighted by observation? Example for 
        more info.
    normalize: bool (optional, default: False)
        Sould the global distribution be a count or a frequency
    
    Example
    -------
    >>> stringlists = ['A', np.nan, 'B;C', 'A;C']
    >>> stringlists_dist(stringlists, weighted=True)
    value  freq
    0     A   1.5
    1     B   0.5
    2     C   1.0
    3   nan   1.0
    >>> stringlists_dist(stringlists, weighted=False)
    value  freq
    0     A     2
    1     B     1
    2     C     2
    3   nan     1
    """
    values = np.unique(append_stringlists(stringlists))
    res = pd.Series(index=values, data=0)
    res["nan"] = 0
    for stringlist in stringlists:
        if isinstance(stringlist, str) and (stringlist != ""):
            stringlist_dist = get_dist_table(
                stringlist.split(";"), normalize=weighted
            )
            stringlist_dist = stringlist_dist.set_index("value").freq
            res[stringlist_dist.index] += stringlist_dist.values
        else:
            res["nan"] += 1
    res = pd.DataFrame({"value": res.index, "freq": res.values})
    if normalize:
        res["freq"] = res["freq"] / np.sum(res["freq"])
    return res
