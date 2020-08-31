# Inspired from Towards data science's article "Time-based cross validation"
# written by Or Herman-Saffar on January 20th, 2020
# https://towardsdatascience.com/time-based-cross-validation-d259b13d42b8

import datetime
from datetime import datetime as dt

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


class TimeBasedCV(object):
    """
    Parameters 
    ----------
    train_period: int
        Number of time units to include in each train set.
    test_period: int
        Number of time units to include in each test set.
    freq: string
        Time unit. Possible values are: days, months, years, weeks, hours,
        minutes, seconds. Possible values designed to be used by 
        dateutil.relativedelta class.
    n_splits: int
        The number of splits to output. Default is None for maximum number of
        time splits with current splitting parameters.
    gap: int
        Number of time units between train and test sets.
    date_column: str
        Column name indicating the date to split on.
    split_dates: list of datetime
        Ignored if n_splits is not None. The dates on which to perform the
        splits.
    method: str, one of "last", "first"
        Where to get the splits. Relevant only if n_splits is not None.
    absolute_index: bool (optional, default: True)
        This splits pandas dataframes because it needs a date column. You can
        chose to get the pandas' index or the absolute index.
    verbose: bool (optional, default: True)
        Whether or not to display the splits.
    """

    def __init__(
        self,
        train_period=12,
        test_period=1,
        freq="months",
        n_splits=None,
        gap=0,
        date_column="date",
        split_dates=None,
        method="last",
        absolute_index=True,
        verbose=True,
    ):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq
        self.n_splits = n_splits
        self.gap = gap
        self.date_column = date_column
        self.split_dates = split_dates
        self.method = method
        self.absolute_index = absolute_index
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into train and test sets.
        
        Parameters 
        ----------
        X: pd.DataFrame
            The data to split. Should have self.date_column in its columns.
        y: 
            always ignored
        groups:
            always ignored
        
        Returns 
        -------
        index: list of tuples of arrays
            List of tuples (train indices, test indices).
        """
        if self.method not in ["last", "first"]:
            raise ValueError("method argument should be one of ['last', 'first']")

        try:
            X[self.date_column]
        except:
            raise KeyError(self.date_column)

        periods = []
        if self.n_splits is None and self.split_dates is not None:
            for split_date in self.split_dates:
                start_train = split_date - eval(f"relativedelta({self.freq}=self.train_period)")
                end_train = start_train + eval(f"relativedelta({self.freq}=self.train_period)")
                start_test = end_train + eval(f"relativedelta({self.freq}=self.gap)")
                end_test = start_test + eval(f"relativedelta({self.freq}=self.test_period)")
                periods.append((start_train, end_train, start_test, end_test))
        else:
            end_test = X[self.date_column].max().date()
            start_test = end_test - eval(f"relativedelta({self.freq}=self.test_period)")
            end_train = start_test - eval(f"relativedelta({self.freq}=self.gap)")
            start_train = end_train - eval(f"relativedelta({self.freq}=self.train_period)")
            periods.append((start_train, end_train, start_test, end_test))
            while start_train > X[self.date_column].min().date():
                end_test = start_test
                start_test = end_test - eval(f"relativedelta({self.freq}=self.test_period)")
                end_train = start_test - eval(f"relativedelta({self.freq}=self.gap)")
                start_train = end_train - eval(f"relativedelta({self.freq}=self.train_period)")
                periods.append((start_train, end_train, start_test, end_test))

            periods = periods[::-1]
            if self.n_splits is not None:
                if self.method == "last":
                    periods = periods[-self.n_splits :]
                elif self.method == "first":
                    periods = periods[: self.n_splits]

        self.n_splits = len(periods)

        index = []
        date_col = X[self.date_column].dt.date
        for start_train, end_train, start_test, end_test in periods:
            train_indices = (date_col >= start_train) & (date_col < end_train)
            test_indices = (date_col >= start_test) & (date_col < end_test)
            if self.absolute_index:
                index.append((np.where(train_indices)[0], np.where(test_indices)[0]))
            else:
                index.append((X[train_indices].index, X[test_indices].index))

            if self.verbose:
                print(
                    f"Train: {start_train} - {end_train}, Test: {start_test} - {end_test} "
                    + f"# {np.sum(train_indices)}, {np.sum(test_indices)}"
                )

        return index

    def get_n_splits(self):
        """Returns the number of splits."""
        return self.n_splits
