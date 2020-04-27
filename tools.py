import datetime
import os
import time

import numpy as np
import pandas as pd


def clear_terminal():
    """ Useful for terminal-interacting scripts """
    os.system("cls" if os.name == "nt" else "clear")


def parent_dir(path):
    """ Extract the parent directory from path """
    if path[-1] == "\\":
        path = path[:-1]
    path = path.split("\\")
    path = path[:-1]
    path = "\\".join(path) + "\\"
    return path


def elapsed_time(t0):
    """ Returns formatted elapsed time since t0 (=time.time()) """
    return str(datetime.timedelta(seconds=time.time() - t0))[:-7]


def partition_n_into_k_subsets(n, k):
    """ Returns a list of k sample sizes who sum up to n, useful for divying up 
    a sample of size n into k subsets if n is not a perfect multiple of k """
    subset_size = int(np.floor(n / k))
    leftover = n - (k * subset_size)
    subsets = [subset_size + 1] * leftover + [subset_size] * (k - leftover)
    return subsets
