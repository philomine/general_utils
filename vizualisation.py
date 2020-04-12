import numpy as np 
import pandas as pd 

import plotly
import plotly.graph_objects as go 
import plotly.express as px

from .data_analysis import dist

def save_fig(fig, filename=None):
    if filename is not None:
        plotly.offline.plot(fig, filename=filename, auto_open=False)

def dist_pie_chart(dist, filename=None, other_cat=True):
    labels = dist.value
    values = dist.freq
    if len(labels) <= 5:
        other_cat = False
    if other_cat:
        total = np.sum(values)
        other_cat = (values / total) < 0.01
        if np.any(other_cat):
            labels = list(labels[~other_cat]) + [f'Other ({np.sum(other_cat)})'] 
            other_values = np.sum(values[other_cat])
            values = list(values[~other_cat]) + [other_values]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent')])
    fig.update_layout(
        annotations=[go.layout.Annotation(
            x=1, y=1,
            showarrow=False,
            text=f"Total number of values: {round(np.sum(values), 0)}",
            xref="paper",
            yref="paper",
            align="right"
        )]
    )
    save_fig(fig, filename=filename)
    return fig

def sample_pie_chart(sample, filename=None, other_cat=True):
    sample_dist = dist(sample)
    return dist_pie_chart(sample_dist, filename=filename, other_cat=other_cat)

def dist_bar_chart(dist, filename=None):
    labels = dist.value
    values = dist.freq
    fig = go.Figure([go.Bar(x=labels, y=values, text=values, textposition='auto')])
    fig.update_layout(
        annotations=[go.layout.Annotation(
            x=1, y=1.05,
            showarrow=False,
            text=f"Total number of values: {round(np.sum(values), 0)}",
            xref="paper",
            yref="paper",
            align="right"
        )]
    )
    save_fig(fig, filename=filename)
    return fig

def sample_bar_chart(sample, filename=None):
    sample_dist = dist(sample)
    return dist_bar_chart(sample_dist, filename=filename)

def scatter_plot(X, labels=None, filename=None):
    if labels:
        color = 'color'
        X = pd.DataFrame(X, columns=['x', 'y'])
        X['color'] = labels 
    else:
        color = None
        X = pd.DataFrame(X, columns=['x', 'y'])
    fig = px.scatter(X, x='x', y='y', color=color)
    save_fig(fig, filename=filename)
    return fig