import os
import re

import plotly
import plotly.express as px
import plotly.graph_objects as go


def _get_scripts(plotly_fig):
    """ Extracts the scripts from a raw plotly html file """
    plotly.offline.plot(plotly_fig, filename="temp.html", auto_open=False)
    f = open("temp.html", "r")
    if f.mode == "r":
        scripts = re.findall(r"<script.*?</script>", f.read(), flags=re.DOTALL)
    f.close()
    os.remove("temp.html")
    return scripts


def _plotly_script():
    """ Extracts the plotly script from a raw plotly html file """
    fig = go.Figure(
        [go.Bar(x=["giraffes", "orangutans", "monkeys"], y=[20, 14, 23])]
    )
    scripts = _get_scripts(fig)
    return f"{scripts[0]}\n{scripts[1]}"


def _extract_plotly_graph(plotly_fig):
    """ Extracts the figure script from a raw plotly html file """
    scripts = _get_scripts(plotly_fig)
    plotly_graph = scripts[2]
    graph_id = re.search(
        r"(?<=getElementById\(\")[a-zA-Z0-9-]*(?=\"\)\))", plotly_graph
    ).group()
    plotly_graph = (
        f'<div id="{graph_id}" class="plotly-graph-div"'
        + f' height="550"></div>\n{plotly_graph}'
    )
    return plotly_graph


def _add_pandas(html_string, df):
    """ Adds raw html code to html_string to display df in html file """
    html_pandas = df.to_html().replace(
        '<table border="1" class="dataframe">',
        '<table class="table table-striped">',
    )
    html_string += f""" 
        {html_pandas}"""
    return html_string


def _add_fig(html_string, fig):
    """ Adds raw html code to html_string to plot fig in html file """
    html_string += f"""
        {_extract_plotly_graph(fig)}"""
    return html_string


def _add_title(html_string, title):
    """ Adds raw html code to html_string to display title in html file """
    html_string += f"""
        <h1>{title}</h1>"""
    return html_string


def _add_subtitle(html_string, subtitle):
    """ Adds raw html code to html_string to display subtitle in html file """
    html_string += f"""
        <h2>{subtitle}</h2>"""
    return html_string


def _add_subsubtitle(html_string, subsubtitle):
    """ Adds raw html code to html_string to display subsubtitle in html file
    """
    html_string += f"""
        <h3>{subsubtitle}</h3>"""
    return html_string


def _add_text(html_string, text):
    """ Adds raw html code to html_string to display text in html file """
    html_string += f"""
        <p>{text}</p>"""
    return html_string


def generate_report(path, contents):
    """ Generate an html file as a report full of plotly figures, tables, 
    titles and texts which are listed in contents list

    Parameters
    ----------
    path: str
        Where to save the html report
    contents: array like of tuple (content_type, content)
        The contents to be displayed in the order you wish to display them. 
        Content-type is one of (title, subtitle, subsubtitle, text, pandas 
        and fig) matching respectively contents of the type (str, str, str, 
        str, pd.DataFrame, plotly figure).
    """
    add = {
        "pandas": _add_pandas,
        "fig": _add_fig,
        "title": _add_title,
        "subtitle": _add_subtitle,
        "subsubtitle": _add_subsubtitle,
        "text": _add_text,
    }

    html_string = f"""
<html>
    <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
        <style>body{{ margin:0 100; background:whitesmoke; }}</style>
    </head>
    <body>
        {_plotly_script()}"""

    for content_type, content in contents:
        html_string = add[content_type](html_string, content)

    html_string += """
    </body>
</html>"""

    f = open(path, "w")
    f.write(html_string)
    f.close()
    return "Report generated"
