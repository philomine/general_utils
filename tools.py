import datetime
import os
import time

import numpy as np
import pandas as pd


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def parent_dir(path):
    if path[-1] == "\\":
        path = path[:-1]
    path = path.split("\\")
    path = path[:-1]
    path = "\\".join(path) + "\\"
    return path


def elapsed_time(t0):
    return str(datetime.timedelta(seconds=time.time() - t0))[:-7]


def partition_n_into_k_subsets(n, k):
    subset_size = int(np.floor(n / k))
    leftover = n - (k * subset_size)
    subsets = [subset_size + 1] * leftover + [subset_size] * (k - leftover)
    return subsets
