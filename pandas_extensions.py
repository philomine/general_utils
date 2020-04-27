import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from .plotly_reporter import generate_report
from .tools import partition_n_into_k_subsets
from .vizualisation import (
    _add_plotly_info,
    _save_plotly_fig,
    _set_plotly_layout,
    dist_table_pie_chart,
    numeric_distribution,
    text_distribution,
    time_series_distribution,
)


def drop_sparse_columns(self, threshold=0.3):
    """ Drops columns with a nan percentage above threshold """
    nb_row, __ = self.shape
    nan_percentage = self.isna().sum() / nb_row
    return self.drop(self.columns[nan_percentage > threshold], axis=1)


setattr(pd.DataFrame, "drop_sparse_columns", drop_sparse_columns)


def full_row_percentage(self):
    """ Returns the the percentage of rows left after a dropna """
    return round(100 * self.dropna().shape[0] / self.shape[0], 2)


setattr(pd.DataFrame, "full_row_percentage", property(full_row_percentage))


def dropna_analysis(self, threshold_precision=0.01):
    """ Returns a plot of the column-dropping threshold choice, helps chosing 
    a threshold to drop sparse columns """
    thresholds = np.arange(0, 1 + threshold_precision, threshold_precision)
    full_row_percentages = []
    for threshold in thresholds:
        temp = self.copy().drop_sparse_columns(threshold=threshold)
        full_row_percentages.append(temp.full_row_percentage)
    results = pd.DataFrame(
        {
            "Maximum nan percentage accepted in columns": thresholds,
            "Full row percentage": full_row_percentages,
        }
    )
    fig = px.line(
        results,
        x="Maximum nan percentage accepted in columns",
        y="Full row percentage",
    )
    return fig


setattr(pd.DataFrame, "dropna_analysis", dropna_analysis)


def column_analysis(self):
    """ Personalised 'describe' """
    analysis = pd.concat(
        [self.nunique(), self.shape[0] - self.isna().sum()], axis=1
    )
    analysis.columns = ["nb_unique", "nb_fill"]
    analysis["nb_rows"] = self.shape[0]
    nb_group = []
    for column in self.columns:
        try:
            __, counts = np.unique(self[column].dropna(), return_counts=True)
        except TypeError:
            __, counts = np.unique(
                self[column].dropna().map(str), return_counts=True
            )
        nb_group.append(np.sum(counts > 1))
    analysis["nb_group"] = nb_group
    analysis["fill_pct"] = 100 * round(analysis.nb_fill / analysis.nb_rows, 4)
    analysis["unique_pct"] = 100 * round(
        analysis.nb_unique / analysis.nb_fill, 4
    )
    analysis["group_pct"] = 100 * round(
        analysis.nb_group / analysis.nb_unique, 4
    )
    analysis["dtypes"] = self.dtypes
    analysis["interesting"] = (analysis.nb_fill != 0) & (
        analysis.nb_unique > 1
    )

    return analysis[
        [
            "dtypes",
            "nb_rows",
            "fill_pct",
            "nb_fill",
            "unique_pct",
            "nb_unique",
            "group_pct",
            "nb_group",
            "interesting",
        ]
    ]


setattr(pd.DataFrame, "column_analysis", column_analysis)


def is_id(self):
    """ Checks wether pd.Series or pd.DataFrame is an id.
    For pd.Series, checks wether it can be used as a primary key. For a 
    pd.DataFrame, checks if all the columns can be used altogether as a primary 
    key. 

    Example
    -------
    >>> series = pd.Series([1, 1, 2])
    >>> series.is_id
    False
    >>> series = pd.Series([1, 2, 3])
    >>> series.is_id
    True
    >>> df = pd.DataFrame({'a': [1, 1, 2]})
    >>> df.is_id
    False
    >>> df = pd.DataFrame({
    >>>     'a': [1, 1, 2],
    >>>     'b': [1, 2, 1],
    >>> })
    >>> df.is_id
    True
    """
    x = self.copy()

    start_size = x.shape[0]
    x = x.dropna().drop_duplicates()
    end_size = x.shape[0]

    return start_size == end_size


setattr(pd.Series, "is_id", property(is_id))
setattr(pd.DataFrame, "is_id", property(is_id))


def divide_dataset(self, nb_div, shuffle=False):
    """ Divides a pandas dataframe into nb_div dataframes, whose sizes are 
    determined by the partition_n_into_k_subsets function """
    n = self.shape[0]
    subset_sizes = partition_n_into_k_subsets(n=n, k=nb_div)

    index = self.index.values.copy()
    if shuffle:
        np.random.shuffle(index)

    start_index = [0] + list(np.cumsum(subset_sizes[:-1]))
    end_index = list(np.cumsum(subset_sizes))

    divs = []
    for beg, end in zip(start_index, end_index):
        divs.append(self.loc[index[beg:end]])

    return divs


setattr(pd.DataFrame, "divide_dataset", divide_dataset)


def nan_figure(self):
    """ Plots a pie chart indicating the number of missing values in a series, 
    for fast viz"""
    nb_empty = np.sum(self.isna())
    nb_filled = np.sum(~self.isna())
    nan_dist = pd.DataFrame(
        {"value": ["Filled", "Empty"], "freq": [nb_filled, nb_empty]}
    )
    fig = dist_table_pie_chart(nan_dist, other_cat=False)
    fig = fig.update_layout(title=f"Share of filled values in {self.name}")
    return fig


setattr(pd.Series, "nan_figure", property(nan_figure))


def plotly_report(self, df_name, path, other_contents=[]):
    """ Generates a plotly report with vizualisation of each columns """
    analysis = self.column_analysis()
    interesting_columns = analysis[analysis.interesting].index.values

    contents = [
        ("title", f"Analysis of dataframe {df_name}"),
        ("pandas", analysis),
    ]
    for column in interesting_columns:
        contents.append(("fig", self[column].plot_distribution()))
    contents += other_contents

    generate_report(path, contents)


setattr(pd.DataFrame, "plotly_report", plotly_report)


def remove_outliers(self, replace_with_nan=False, bound=1.5):
    Q1 = np.quantile(self.dropna(), 0.25)
    Q3 = np.quantile(self.dropna(), 0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        raise Warning(
            "The interquartile range is 0, this will consider all values "
            + "different from 0 to be outliers."
        )

    min_bound = Q1 - bound * IQR
    max_bound = Q3 + bound * IQR

    outlier = (self < min_bound) | (self > max_bound)

    sample = self.copy()
    if replace_with_nan:
        sample[outlier] = np.nan
    else:
        sample = sample[~outlier]

    return sample


setattr(pd.Series, "remove_outliers", remove_outliers)


def _add_sample_info(fig, sample_size, nb_nan):
    text = (
        f"Number of values: {sample_size}"
        + "<br>"
        + f"Filled: {sample_size - nb_nan} "
        + "({round(100 * (sample_size - nb_nan) / sample_size, 2)}%)"
        + "<br>"
        + f"Empty: {nb_nan} ({round(100 * nb_nan / sample_size, 2)}%)"
    )
    fig = _add_plotly_info(fig, x=1, y=1.2, text=text)
    return fig


def _reset_kwargs(kwargs):
    kwargs["filename"] = None
    kwargs["sample_info"] = False
    return kwargs


def plot_distribution(
    self, sample_info=True, filename=None, plot_type=None, **kwargs
):
    """ Plots the distribution of a pd.Series depending on its type
    Relies on the vizualisation module's plotting functions and makes it easy 
    to handle missing values (adds missing values info in sample info).
    """
    if self.name is None:
        self.name = "unknown sample"

    dtype_distribution = {
        "object": text_distribution,
        "str": text_distribution,
        "datetime64[ns]": time_series_distribution,
        "int64": numeric_distribution,
        "float64": numeric_distribution,
    }
    if plot_type is None:
        plot_type = str(self.dtype)

    dtype_distribution = dtype_distribution[plot_type]
    fig = dtype_distribution(
        self.copy().dropna(),
        title=f"Distribution of {self.name}",
        **_reset_kwargs(kwargs),
    )
    if sample_info:
        fig = _add_sample_info(fig, self.shape[0], np.sum(self.isna()))
    _save_plotly_fig(fig, filename=filename)
    return fig


setattr(pd.Series, "plot_distribution", plot_distribution)
