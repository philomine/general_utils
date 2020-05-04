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


def _save_results(results, save_path):
    """ Saves results df to save_path (without being interrupted) """
    try:
        clear_terminal()
        print("Saving...")
        results.to_excel(save_path, index=False)
        clear_terminal()
        print(
            f"You exited the labeling process. Your labeling has been "
            + f"saved to {save_path}."
        )
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
        results = pd.DataFrame(columns=df.columns + ["labels"])
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
