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


def continuous_to_binary(x, threshold):
    """Changes values in x from continuous to binary (0 or 1) based on
    threshold. Values <= to threshold will be 0 and values > to threshold will
    be 1.
    
    Parameters
    ----------
    x: array like of continuous values
        Will be cast to np.array for ease of slice.
    threshold: numeric
        Should be comparable to values in x.
    
    Returns
    -------
    x: np.array
        The transformed x input.

    Example
    -------
    >>> x = [1.2, 0.3, 4.5, -1]
    >>> continuous_to_binary(x, 1)
    np.array([1, 0, 1, 0])
    """
    res = np.array(x).copy()
    res[x <= threshold] = 0
    res[x > threshold] = 1
    return res


def split_X_y(data, target, to_numpy=True, return_feature_names=False):
    """Splits data into X and y with y being the target column. Expecting
    pd.DataFrame.
    
    Parameters
    ----------
    data: pd.DataFrame
        Dataframe to split into X and y, should have target in columns.
    target: string
        Should be in data.columns.
    to_numpy: bool (optional, default: True)
        Wheter to turn the results to numpy arrays.
    return_feature_names: bool (optional, default: False)
        Wheter to return X, y, feature_names or simply X, y. Useful when
        to_numpy=True.
    
    Returns
    -------
    X: pd.DataFrame
        data input without target column.
    y: pd.Serie
        The target column taken in data input.
    feature_names: list of str (optional, depends on the return_feature_names param)
        The names of the features in X.
    """
    X = data[[col for col in data.columns if col != target]].copy()
    y = data[target].copy()

    if return_feature_names:
        feature_names = X.columns.copy()

    if to_numpy:
        X = X.to_numpy()
        y = y.to_numpy().flatten()

    if return_feature_names:
        res = X, y, feature_names
    else:
        res = X, y

    return res


def _save_results(results, save_path):
    """ Saves results df to save_path (without being interrupted) """
    try:
        clear_terminal()
        print("Saving...")
        results.to_excel(save_path, index=False)
        clear_terminal()
        print(f"You exited the labeling process. Your labeling has been " + f"saved to {save_path}.")
    except KeyboardInterrupt:
        _save_results(results, save_path)


def df_explorer(df, save_path=None, label=True):
    """ Explore any pd.DataFrame df and save your comments to save_path.
    You can chose to either explore the dataset (label=False) or explore and
    save your comments (label=True).
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe you want to explore. The function iterates over the rows
        and the row's content is displayed in console until you comment or
        simply go to next.
    save_path: str or None (optional, default:None)
        The path where the comments are saved (if label=True). Can't be None if
        label=True.
    label: bool (optional, default:True)
        Wether to label the explored rows. Set to False if you just want to
        peak at the data, to True if you want to peak and sometimes (or always)
        add a label/comment/whatever.
    """
    try:
        results = pd.DataFrame()
        for __, row in df.iterrows():
            # Printing instructions
            clear_terminal()
            for field_name, value in row.iteritems():
                print(f"{field_name}: {value}")

            # Prompting for label
            print("")
            print("Exit: press ctrl-C (the exiting may take a few seconds)")
            if label:
                row["label"] = input("Label: ")
            else:
                input("Press enter for next observation.")

            # Saving label
            if label:
                results = results.append(row, ignore_index=True,)
    except KeyboardInterrupt:
        if label:
            _save_results(results, save_path + "/temp.xlsx")
