import numpy as np
import pandas as pd
import pickle

import plotly.express as px

from .data_analysis import string_format, dist
from .plotly_reporter import generate_report
from .vizualisation import dist_pie_chart

import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

def drop_sparse_columns(self, threshold=0.3):
    """ Drops columns with a nan percentage above threshold """
    nb_row, __ = self.shape
    nan_percentage = self.isna().sum() / nb_row
    return self.drop(self.columns[nan_percentage > threshold], axis=1)
setattr(pd.DataFrame, 'drop_sparse_columns', drop_sparse_columns)

def full_row_percentage(self):
    """ Returns the number of rows and the percentage of rows left after a dropna """
    return self.dropna().shape[0] / self.shape[0]
setattr(pd.DataFrame, 'full_row_percentage', property(full_row_percentage))

def dropna_analysis(self, threshold_precision=0.01):
    """ Returns a plot of the column-dropping threshold choice 
    x axis: Percentage of maximum tolerated nan in columns
    y axis: Percentage of full rows left after dropping these columns
    """
    x = np.arange(0, 1 + threshold_precision, threshold_precision)
    full_row_percentage = []
    temp = self.copy()
    for threshold in x[::-1]:
        temp = temp.drop_empty_columns(threshold=threshold)
        full_row_percentage.insert(0, temp.shape_and_full_row_percentage()[1])
    results = pd.DataFrame({'Maximum nan percentage accepted in columns': x, 'Full row percentage': full_row_percentage})
    fig = px.line(results, x='Maximum nan percentage accepted in columns', y='Full row percentage')
    fig.show()
setattr(pd.DataFrame, 'dropna_analysis', dropna_analysis)

def column_analysis(self, detail=True):
    analysis = pd.concat([self.nunique(), self.shape[0] - self.isna().sum()], axis=1)
    analysis.columns = ['nb_unique', 'nb_fill']
    analysis['nb_rows'] = self.shape[0]
    nb_group = []
    for column in self.columns:
        try:
            __, counts = np.unique(self[column].dropna(), return_counts=True)
        except TypeError:
            __, counts = np.unique([str(val) for val in self[column].dropna()], return_counts=True)
        nb_group.append(np.sum(counts > 1))
    analysis['nb_group'] = nb_group
    analysis['fill_pct'] = 100 * round(analysis.nb_fill / analysis.nb_rows, 4)
    analysis['unique_pct'] = 100 * round(analysis.nb_unique / analysis.nb_fill, 4)
    analysis['group_pct'] = 100 * round(analysis.nb_group / analysis.nb_unique, 4)
    analysis['dtypes'] = self.dtypes
    analysis['interesting'] = (analysis.nb_fill != 0) & (analysis.nb_unique > 1)

    if detail:
        return_columns = [
            'dtypes', 'nb_rows', 'fill_pct', 'nb_fill', 'unique_pct', 
            'nb_unique', 'group_pct', 'nb_group', 'interesting',
        ]
    else:
        return_columns = [
            'dtypes', 'fill_pct', 'unique_pct', 'group_pct', 'interesting',
        ]
    
    return analysis[return_columns]
setattr(pd.DataFrame, 'column_analysis', property(column_analysis))

def values_stats(self):
    nan_count = np.sum(self.isna())
    try:
        unique, counts = np.unique(self.dropna(), return_counts=True)
    except TypeError:
        unique, counts = np.unique([str(val) for val in self.dropna()], return_counts=True)
    stats = pd.DataFrame({'unique': unique, 'count': counts})
    stats = stats.append(pd.DataFrame({'unique': [np.nan], 'count': [nan_count]}), ignore_index=True)
    stats['pct'] = stats['count'] / np.sum(stats['count'])
    return stats
setattr(pd.Series, 'values_stats', values_stats)

def string_formats(self):
    res = []
    for val in self:
        res.append(string_format(val))
    return pd.Series(res)
setattr(pd.Series, 'string_formats', string_formats)

def is_id(self):
    series = self.copy()

    start_size = series.size
    series = series.dropna().drop_duplicates()
    end_size = series.size

    return start_size == end_size
setattr(pd.Series, 'is_id', property(is_id))

def is_id(self, columns):
    df = self.copy()

    start_size = df.shape[0]
    df = df.dropna(subset=columns).drop_duplicates(columns)
    end_size = df.shape[0]

    return start_size == end_size 
setattr(pd.DataFrame, 'is_id', is_id)





def partition_n_into_k_subsets(n, k):
    subset_size = int(np.floor(n / k))
    leftover = n - (k * subset_size)
    subsets = [subset_size + 1] * leftover + [subset_size] * (k - leftover)
    return subsets

def divide_dataset(df, nb_div, shuffle=False):
    n = df.shape[0]
    subset_sizes = partition_n_into_k_subsets(n=n, k=nb_div)
    
    index = df.index.values.copy()
    if shuffle:
        np.random.shuffle(index)
    
    start_index = [0] + list(np.cumsum(subset_sizes[:-1]))
    end_index = list(np.cumsum(subset_sizes))
    
    divs = []
    for beg, end in zip(start_index, end_index):
        divs.append(df.loc[index[beg:end]])
    
    return divs




def time_series_distribution(sample, sample_name=None, sample_size=300, freq='D'):
    """ Plots a distribution of a timestamp sample 
    
    Parameters
    ----------
    sample : array-like of pd.Timestamp
        The dates of which to plot the distribution.
    
    sample_size: int (default: 300)
        Number of dates to plot (if too big, the bars are 
        too slim to see).
    
    freq: pd.Timestamp's freq
        You can group the dates in freq: 'W' for weeks of year,
        'M' for months, etc. This allows you to go into less 
        detail but see all values globally if your sample is too 
        large.
    """
    if freq != 'D':
        def floor_timestamp(timestamp):
            return timestamp.to_period(freq).to_timestamp()
        sample = list(map(floor_timestamp, sample))
    
    sample_distribution = dist(sample)
    sample_distribution = pd.DataFrame({'value': pd.date_range(
        start=np.min(sample_distribution.value),
        end=np.max(sample_distribution.value),
        freq=freq)
    }).join(sample_distribution.set_index('value'), on='value').fillna(0)
    
    if sample_size > sample_distribution.shape[0]:
        sample_size = sample_distribution.shape[0]
        
    if sample_distribution.shape[0] > sample_size:
        sample_distributions = divide_dataset(
            sample_distribution, 
            int(np.ceil(sample_distribution.shape[0] / sample_size)),
        )
        fig = go.Figure()
        for i, data in enumerate(sample_distributions):
            if np.any([val != 0 for val in data.freq]):
                fig.add_trace(go.Bar(visible=False, x=data.value, y=data.freq))
        fig.data[-1].visible = True
        steps = []
        for i in range(len(fig.data)):
            visibility = [False] * len(fig.data)
            visibility[i] = True
            steps.append(dict(
                method="restyle",
                args=["visible", visibility],
                label=str(i),
            ))

        sliders = [dict(
            active=len(fig.data)-1,
            x=0, y=-0.2,
            currentvalue={"prefix": "Slice: "},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sample_distribution.value,
            y=sample_distribution.freq,
        ))
    
    if sample_name is not None:
        fig.update_layout(title=sample_name)
    
    return fig

def numeric_distribution(sample, sample_name=None, sample_size=50):
    nb_unique = len(np.unique(sample))
    if nb_unique <= 20:
        fig = text_distribution(
            sample.astype(str), 
            sample_name=sample_name, 
        )
    else:
        sample_range = int(np.ceil(np.max(sample) - np.min(sample)))
        if sample_range > sample_size:
            fig = go.Figure(data=[go.Histogram(x=sample, nbinsx=sample_size)])
        else:
            fig = go.Figure(data=[go.Histogram(x=sample)])
        fig.update_layout(yaxis_type="log")
        if sample_name is not None:
            fig.update_layout(title=sample_name)
    return fig

def text_distribution(sample, sample_name=None, sample_size=20):
    sample = sample.astype(str)
    sample_distribution = dist(sample)
    if sample_distribution.shape[0] > sample_size:
        formats = sample.map(string_format)
        format_distribution = dist(formats)
        if format_distribution.shape[0] > sample_size:
            lengths = pd.Series([len(val) for val in sample])
            fig = numeric_distribution(lengths, sample_name=f"Lenght of {sample_name}")
        else:
            fig = text_distribution(formats, sample_name=f'Format of {sample_name}')
    else:
        fig = go.Figure(data=[go.Bar(
            x=sample_distribution.value,
            y=sample_distribution.freq,
        )])
        if sample_name is not None:
            fig.update_layout(title=sample_name)
    fig.update_layout(yaxis_type="log")
    return fig

def distribution(self):
    dtype_distribution = {
        'object': text_distribution,
        'datetime64[ns]': time_series_distribution,
        'int64': numeric_distribution,
        'float64': numeric_distribution,
    }
    dtype_distribution = dtype_distribution[str(self.dtype)]
    fig = dtype_distribution(self.dropna(), sample_name=self.name)
    fig.update_layout(
        annotations=[go.layout.Annotation(
            x=1, y=1.2,
            showarrow=False,
            text=f"Number of values: {self.shape[0]}" + "<br>" + \
                f"Filled: {self.dropna().shape[0]}" + "<br>" + \
                f"Empty: {self.shape[0] - self.dropna().shape[0]}",
            xref="paper",
            yref="paper",
            align="right"
        )]
    )
    return fig
setattr(pd.Series, 'distribution', property(distribution))

def nan_distribution(self):
    nb_empty = np.sum(self.isna())
    nb_filled = np.sum(~self.isna())
    sample_dist = pd.DataFrame({
        'value': ['Filled', 'Empty'], 'freq': [nb_filled, nb_empty]
    })
    fig = dist_pie_chart(sample_dist, other_cat=False)
    fig = fig.update_layout(title=f'Share of filled values in {self.name}')
    return fig
setattr(pd.Series, 'nan_distribution', property(nan_distribution))


def plotly_report(self, df_name, path, other_contents=[]):
    analysis = self.column_analysis
    interesting_columns = analysis[analysis.interesting].index.values

    contents = [
        ('title', f'Analysis of dataframe {df_name}'),
        ('pandas', self.column_analysis),
    ]
    for column in interesting_columns:
        contents.append(('fig', self[column].distribution))
    contents += other_contents
    
    generate_report(path, contents)
setattr(pd.DataFrame, 'plotly_report', plotly_report)