import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go

from data_analysis import get_dist_table, get_string_format


def _set_plotly_layout(fig, title=None, log_scale=True):
    if title is not None:
        fig.update_layout(title=title)
    if log_scale:
        fig.update_layout(yaxis_type="log")
    return fig


def _add_plotly_info(fig, x, y, text, align="right"):
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
    text = f"Number of values: {sample_size}"
    fig = _add_plotly_info(fig, x=1, y=1.08, text=text)
    return fig


def _add_num_info(fig, num_sample):
    """ Adds information about numeric sample on fig 
    num_sample must not have null values
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
    if filename is not None:
        plotly.offline.plot(fig, filename=filename, auto_open=False)


def reset_kwargs(kwargs):
    kwargs["title"] = None
    kwargs["filename"] = None
    kwargs["sample_info"] = False
    kwargs["log_scale"] = False
    kwargs["num_info"] = False
    return kwargs


def sample_pie_chart(
    sample, title=None, filename=None, sample_info=True, **kwargs
):
    sample_dist_table = get_dist_table(sample)

    # Plotting the figure
    fig = dist_table_pie_chart(sample_dist_table, **reset_kwargs(kwargs))

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
    print(other_cat)
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
    num_info=False,
    log_scale=True,
    **kwargs,
):
    sample_dist_table = get_dist_table(sample)

    # Plotting the figure
    fig = dist_table_bar_chart(sample_dist_table, **reset_kwargs(kwargs))

    # Layout and saving parameters
    fig = _set_plotly_layout(fig, title=title, log_scale=log_scale)
    if sample_info:
        fig = _add_sample_info(fig, len(sample))
    if num_info:
        fig = _add_num_info(fig, [float(val) for val in sample])
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


def scatter_plot(x, y, labels=None, filename=None):
    if labels:
        color = "labels"
        X = pd.DataFrame({"x": x, "y": y})
        X["labels"] = labels
    else:
        color = None
        X = pd.DataFrame({"x": x, "y": y})
    fig = px.scatter(X, x="x", y="y", color=color)
    _save_plotly_fig(fig, filename=filename)
    return fig


def time_series_distribution(
    sample,
    title=None,
    filename=None,
    sample_info=True,
    log_scale=True,
    nbins=300,
    time_freq="D",
    **kwargs,
):
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
    nbins=300,
    **kwargs,
):
    if len(np.unique(sample)) <= 20:
        fig = text_distribution(
            [str(val) for val in sample], **reset_kwargs(kwargs)
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
    sample_distribution = get_dist_table(sample)
    if sample_distribution.shape[0] > nbins:
        formats = list(map(lambda x: get_string_format(x), sample))
        format_distribution = get_dist_table(formats)
        if format_distribution.shape[0] > nbins:
            lengths = pd.Series([len(val) for val in sample])
            fig = numeric_distribution(lengths)
        else:
            fig = text_distribution(formats, **reset_kwargs(kwargs))
    else:
        if text_as_pie:
            fig = sample_pie_chart(sample, **reset_kwargs(kwargs))
        else:
            fig = sample_bar_chart(sample, **reset_kwargs(kwargs))
    # Layout and saving parameters
    fig = _set_plotly_layout(
        fig, title=title, log_scale=(log_scale and not text_as_pie)
    )
    if sample_info:
        fig = _add_sample_info(fig, sample_size=len(sample))
    _save_plotly_fig(fig, filename=filename)
    return fig
