import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go

from .data_analysis import get_dist_table, get_sample, get_string_format


def _set_plotly_layout(fig, title=None, log_scale=False):
    """ Sets two major layout info for plotly plots (title and log scale) 
    
    Parameters
    ----------
    fig: plotly figure
        The figure for which to set the layout
    title: str or None (optional, default: None)
        The title of the figure, None won't display any title
    log_scale: bool {True or False} (optional, default: False)
        Sets the y axis to log scale
    """
    if title is not None:
        fig.update_layout(title=title)
    if log_scale:
        fig.update_layout(yaxis_type="log")
    return fig


def _add_plotly_info(fig, x, y, text, align="right"):
    """ Adds an info on the plot with an annotation 

    Parameters
    ----------
    fig: plotly figure
        The figure on which to add the info
    x: float
        The x position of the annotation (x=0 matches left of figure)
    y: float
        The y position of the annotation (y=0 matches the bottom of figure)
    text: str
        The text that will be displayed
    align: str {"left", "right"} (optional, default: "right")
        The alignment of the text
    """
    fig.add_annotation(
        x=x,
        y=y,
        showarrow=False,
        text=text,
        xref="paper",
        yref="paper",
        align=align,
    )
    return fig


def _add_sample_info(fig, sample_size):
    """ Adds the sample size on the plot 

    Parameters
    ----------
    fig: plotly figure
        The figure on which to add the sample size info
    sample_size: int
        The sample size that will be displayed on the plot
    """
    text = f"Number of values: {sample_size}"
    fig = _add_plotly_info(fig, x=1, y=1.08, text=text)
    return fig


def _add_num_info(fig, num_sample):
    """ Adds information about numeric sample on fig 
    num_sample must not have null values. Numeric info are Q1, Q3, median, min, 
    max and mean.

    Parameters
    ----------
    fig: plotly figure
        The figure on which to add the numeric info
    num_sample: array-like of numeric values
        The numeric sample for which to compute the information. Must not have 
        null values.
    """
    text = (
        f"Q1: {round(np.quantile(num_sample, 0.25), 2)}"
        + "<br>"
        + f"min: {round(np.min(num_sample), 2)}"
    )
    fig = _add_plotly_info(fig, x=0, y=1.15, text=text, align="left")
    text = (
        f"med: {round(np.quantile(num_sample, 0.5), 2)}"
        + "<br>"
        + f"mean: {round(np.mean(num_sample), 2)}"
    )
    fig = _add_plotly_info(fig, x=0.15, y=1.15, text=text, align="left")
    text = (
        f"Q3: {round(np.quantile(num_sample, 0.75), 2)}"
        + "<br>"
        + f"max: {round(np.max(num_sample), 2)}"
    )
    fig = _add_plotly_info(fig, x=0.4, y=1.15, text=text, align="left")
    return fig


def _save_plotly_fig(fig, filename=None):
    """ Saves the figure to filename 
    
    Parameters
    ----------
    fig: plotly figure
        The figure to save
    filename: str or None (optional, default: None)
        The path in which to save the plotly figure. If None, the figure isn't 
        saved. Should be an html file.
    """
    if filename is not None:
        plotly.offline.plot(fig, filename=filename, auto_open=False)


def _reset_kwargs(kwargs):
    """ Resets the kwargs of the plotting API.
    The plotting API calls and is being called back and forth. Functions rely 
    on each other. To allow passing arguments to other used functions, **kwargs 
    were added. But to keep underlying functions to set some layout on your 
    figure, you can reset the kwargs when calling it: it will then plot a naked 
    figure, leaving the primary function called to set the layout.

    Parameters
    ----------
    kwargs: dict
        The kwargs received and to be reset
    """
    kwargs["title"] = None
    kwargs["filename"] = None
    kwargs["sample_info"] = False
    kwargs["log_scale"] = False
    kwargs["num_info"] = False
    return kwargs


def _str_is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def sample_pie_chart(
    sample, title=None, filename=None, sample_info=True, **kwargs
):
    """ Plots a pie chart from a sample, best suitable for attributes sample, 
    sample should not have any null values

    Parameters
    ----------
    sample: array-like of values
        The sample for which you want your pie chart. Better be an 'attribute'
        sample, meaning a non-numerical sample, with not many values - for 
        vizualisation comfort. Must not have any null values.
    title: str or None (optional, default: None)
        Title of the plot. Set to None for no title.
    filename: str or None (optional, default: None)
        The path where to save the plot. Set to None to not save the plot.
    sample_info: bool (optional, default: True)
        Wether to write the sample size (set to True because it's usually 
        useful to have this basic info).
    """
    sample_dist_table = get_dist_table(sample)

    # Plotting the figure
    fig = dist_table_pie_chart(sample_dist_table, **_reset_kwargs(kwargs))

    # Layout and saving parameters
    fig = _set_plotly_layout(fig, title=title, log_scale=False)
    if sample_info:
        fig = _add_sample_info(fig, len(sample))
    _save_plotly_fig(fig, filename=filename)
    return fig


def dist_table_pie_chart(
    sample_dist_table,
    title=None,
    filename=None,
    sample_info=True,
    other_cat=True,
    **kwargs,
):
    """ Plots a pie chart from a dist table
    A dist table is a pd.DataFrame with two columns: 
        - value: list of values
        - freq: frequency of those values
    Dist tables can be derived from samples thanks to the 
    data_analysis.get_dist_table function. Pie charts are more suitable for 
    'attributes' samples.

    Parameters
    ----------
    sample_dist_table: pd.DataFrame (dist table format)
        The dist table of the sample for which to plot the pie chart.
    title: str or None (optional, default: None)
        Title of the plot. Set to None for no title.
    filename: str or None (optional, default: None)
        The path where to save the plot. Set to None to not save the plot.
    sample_info: bool (optional, default: True)
        Wether to write the sample size (set to True because it's usually 
        useful to have this basic info).
    other_cat: bool (optional, default: True)
        Wether to join all 'small' categories under one 'Other' category.
    """
    labels = sample_dist_table.value.map(lambda x: str(x))
    values = sample_dist_table.freq

    # Creating an 'Other' category if there are several small categories
    if other_cat and len(labels) > 5:
        total = np.sum(values)
        other_cat = (values / total) < 0.01
        if np.sum(other_cat) > 1:
            labels = list(labels[~other_cat]) + [
                f"Other ({np.sum(other_cat)})"
            ]
            other_values = np.sum(values[other_cat])
            values = list(values[~other_cat]) + [other_values]

    # Plotting the figure
    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, textinfo="label+percent")]
    )

    # Layout and saving parameters
    fig = _set_plotly_layout(fig, title=title, log_scale=False)
    if sample_info:
        fig = _add_sample_info(fig, sample_size=int(np.sum(values)))
    _save_plotly_fig(fig, filename=filename)
    return fig


def sample_bar_chart(
    sample,
    title=None,
    filename=None,
    sample_info=True,
    log_scale=True,
    **kwargs,
):
    """ Plots a bar chart from a sample, best suitable for 'attributes' sample, 
    sample should not have any null values

    Parameters
    ----------
    sample: array-like of values
        The sample for which you want your bar chart. Better be an 'attribute'
        sample, meaning a non-numerical sample, with not many values - for 
        vizualisation comfort. Must not have any null values.
    title: str or None (optional, default: None)
        Title of the plot. Set to None for no title.
    filename: str or None (optional, default: None)
        The path where to save the plot. Set to None to not save the plot.
    sample_info: bool (optional, default: True)
        Wether to write the sample size (set to True because it's usually 
        useful to have this basic info).
    log_scale: bool (optional, default: True)
        Wether to set the y axis scale to be logarithmic.
    """
    # Plotting the figure
    sample_dist_table = get_dist_table(sample)
    fig = dist_table_bar_chart(sample_dist_table, **_reset_kwargs(kwargs))

    # Layout and saving parameters
    fig = _set_plotly_layout(fig, title=title, log_scale=log_scale)
    if sample_info:
        fig = _add_sample_info(fig, len(sample))
    _save_plotly_fig(fig, filename=filename)
    return fig


def dist_table_bar_chart(
    sample_dist_table,
    title=None,
    filename=None,
    sample_info=True,
    log_scale=True,
    **kwargs,
):
    """ Plots a bar chart from a dist table
    A dist table is a pd.DataFrame with two columns: 
        - value: list of values
        - freq: frequency of those values
    Dist tables can be derived from samples thanks to the 
    data_analysis.get_dist_table function. Bar charts are more suitable for 
    'attributes' samples.

    Parameters
    ----------
    sample_dist_table: pd.DataFrame (dist table format)
        The dist table of the sample for which to plot the bar chart.
    title: str or None (optional, default: None)
        Title of the plot. Set to None for no title.
    filename: str or None (optional, default: None)
        The path where to save the plot. Set to None to not save the plot.
    sample_info: bool (optional, default: True)
        Wether to write the sample size (set to True because it's usually 
        useful to have this basic info).
    log_scale: bool (optional, default: True)
        Wether to set the y axis scale to be logarithmic.
    """
    labels = sample_dist_table.value
    values = sample_dist_table.freq

    # Plotting the figure
    fig = go.Figure(
        [go.Bar(x=labels, y=values, text=values, textposition="auto")]
    )

    # Layout and saving parameters
    fig = _set_plotly_layout(fig, title=title, log_scale=log_scale)
    if sample_info:
        fig = _add_sample_info(fig, sample_size=int(np.sum(values)))
    _save_plotly_fig(fig, filename=filename)
    return fig


def scatter_plot(
    x, y, labels=None, title=None, filename=None, sample_info=True
):
    """ Plots a scatter plot with (x, y) point coordinates 
    Parameters x and y should have the same length.
    
    Parameters
    ----------
    x: array like of float
        List of the x-coordinates of the points
    y: array like of float
        List of the y-coordinates of the points
    labels: array like of str or None (optional, default: None)
        List of the points' label (if points have labels)
    title: str or None (optional, default: None)
        Title of the plot. Set to None for no title.
    filename: str or None (optional, default: None)
        The path where to save the plot. Set to None to not save the plot.
    sample_info: bool (optional, default: True)
        Wether to write the sample size (set to True because it's usually 
        useful to have this basic info).
    """
    if labels:
        color = "labels"
        X = pd.DataFrame({"x": x, "y": y})
        X["labels"] = labels
    else:
        color = None
        X = pd.DataFrame({"x": x, "y": y})
    fig = px.scatter(X, x="x", y="y", color=color)
    fig = _set_plotly_layout(fig, title=title, log_scale=False)
    if sample_info:
        fig = _add_sample_info(fig, sample_size=len(x))
    _save_plotly_fig(fig, filename=filename)
    return fig


def time_series_distribution(
    sample,
    title=None,
    filename=None,
    sample_info=True,
    log_scale=False,
    nbins=200,
    time_freq="D",
    **kwargs,
):
    """ Plots a distribution of a timestamp sample 
    
    Parameters
    ----------
    sample : array-like of pd.Timestamp
        The dates of which to plot the distribution.
    title: str or None (optional, default: None)
        Title of the plot. Set to None for no title.
    filename: str or None (optional, default: None)
        The path where to save the plot. Set to None to not save the plot.
    sample_info: bool (optional, default: True)
        Wether to write the sample size (set to True because it's usually 
        useful to have this basic info).
    log_scale: bool (optional, default: True)
        Wether to set the y axis scale to be logarithmic.
    nbins: int (optional, default: 200)
        Number of dates to plot (if too big, the bars are too slim to see).
    time_freq: pd.Timestamp's freq (optional, default: "D")
        You can group the dates in freq: 'W' for weeks of year, 'M' for months, 
        etc. This allows you to go into less detail but see all values globally 
        if your sample is too large.
    """
    if time_freq != "D":
        sample = list(
            map(
                lambda timestamp: timestamp.to_period(
                    time_freq
                ).to_timestamp(),
                sample,
            )
        )

    sample_distribution = get_dist_table(sample)
    sample_distribution = (
        pd.DataFrame(
            {
                "value": pd.date_range(
                    start=np.min(sample_distribution.value),
                    end=np.max(sample_distribution.value),
                    freq=time_freq,
                )
            }
        )
        .join(sample_distribution.set_index("value"), on="value")
        .fillna(0)
    )

    if nbins > sample_distribution.shape[0]:
        nbins = sample_distribution.shape[0]

    if sample_distribution.shape[0] > nbins:
        sample_distributions = sample_distribution.divide_dataset(
            int(np.ceil(sample_distribution.shape[0] / nbins)),
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
            steps.append(
                dict(
                    method="restyle",
                    args=["visible", visibility],
                    label=str(i),
                )
            )

        sliders = [
            dict(
                active=len(fig.data) - 1,
                x=0,
                y=-0.2,
                currentvalue={"prefix": "Slice: "},
                steps=steps,
            )
        ]

        fig.update_layout(sliders=sliders)
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=sample_distribution.value, y=sample_distribution.freq)
        )

    # Layout and saving parameters
    fig = _set_plotly_layout(fig, title=title, log_scale=log_scale)
    if sample_info:
        fig = _add_sample_info(fig, sample_size=len(sample))
    _save_plotly_fig(fig, filename=filename)

    return fig


def numeric_distribution(
    sample,
    title=None,
    filename=None,
    sample_info=True,
    num_info=True,
    log_scale=True,
    nbins=200,
    **kwargs,
):
    """ Plots the distribution of a numeric sample with no null values
    
    Parameters
    ----------
    sample : array-like of numeric values
        The numeric values for which to plot the distribution. Should not have 
        any null values.
    title: str or None (optional, default: None)
        Title of the plot. Set to None for no title.
    filename: str or None (optional, default: None)
        The path where to save the plot. Set to None to not save the plot.
    sample_info: bool (optional, default: True)
        Wether to write the sample size (set to True because it's usually 
        useful to have this basic info).
    num_info: bool (optional, default: False)
        Wether to write the numerical info on the plot.
    log_scale: bool (optional, default: True)
        Wether to set the y axis scale to be logarithmic.
    nbins: int (optional, default: 200)
        Number of bins to plot (if too big, the bars are too slim to see).
    """
    if len(np.unique(sample)) <= 20:
        fig = text_distribution(
            [str(val) for val in sample], **_reset_kwargs(kwargs)
        )
    else:
        sample_range = int(np.ceil(np.max(sample) - np.min(sample)))
        if sample_range > nbins:
            fig = go.Figure(data=[go.Histogram(x=sample, nbinsx=nbins)])
        else:
            fig = go.Figure(data=[go.Histogram(x=sample)])
    # Layout and saving parameters
    fig = _set_plotly_layout(fig, title=title, log_scale=log_scale)
    if sample_info:
        fig = _add_sample_info(fig, sample_size=len(sample))
    if num_info:
        fig = _add_num_info(fig, [float(val) for val in sample])
    _save_plotly_fig(fig, filename=filename)
    return fig


def text_distribution(
    sample,
    title=None,
    filename=None,
    sample_info=True,
    log_scale=True,
    text_as_pie=True,
    nbins=20,
    **kwargs,
):
    """ Plots the distribution of an 'attribute' sample with no null values

    Parameters
    ----------
    sample : array-like of attributes values
        The attributes values for which to plot the distribution. Should not 
        have any null values.
    title: str or None (optional, default: None)
        Title of the plot. Set to None for no title.
    filename: str or None (optional, default: None)
        The path where to save the plot. Set to None to not save the plot.
    sample_info: bool (optional, default: True)
        Wether to write the sample size (set to True because it's usually 
        useful to have this basic info).
    log_scale: bool (optional, default: True)
        Wether to set the y axis scale to be logarithmic.
    text_as_pie: bool (optional, default: True)
        Attributes values are plotted either as a pie or bar chart.
    nbins: int (optional, default: 20)
        Number of maximum number of values to plot. For attributes values, it 
        quickly becomes unreadable. If the number of unique values is too big, 
        We'll evaluate if there's a limited number of formats for the sample, 
        and if not we'll resort to plotting the length of the different values.
    """
    sample_distribution = get_dist_table(sample)
    if sample_distribution.shape[0] > nbins:
        formats = list(map(lambda x: get_string_format(x), sample))
        format_distribution = get_dist_table(formats)
        if format_distribution.shape[0] > nbins:
            lengths = pd.Series([len(val) for val in sample])
            fig = numeric_distribution(lengths)
        else:
            fig = text_distribution(formats, **_reset_kwargs(kwargs))
    else:
        if text_as_pie:
            fig = sample_pie_chart(sample, **_reset_kwargs(kwargs))
        else:
            fig = sample_bar_chart(sample, **_reset_kwargs(kwargs))
    # Layout and saving parameters
    fig = _set_plotly_layout(
        fig, title=title, log_scale=(log_scale and not text_as_pie)
    )
    if sample_info:
        fig = _add_sample_info(fig, sample_size=len(sample))
    _save_plotly_fig(fig, filename=filename)
    return fig
