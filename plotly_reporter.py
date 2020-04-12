import plotly 
import plotly.graph_objects as go 
import plotly.express as px 

import os 
import re 

def _get_scripts(plotly_fig):
    plotly.offline.plot(plotly_fig, filename='temp.html', auto_open=False)
    f = open('temp.html', 'r')
    if f.mode == 'r':
        scripts = re.findall(r"<script.*?</script>", f.read(), flags=re.DOTALL)
    f.close()
    os.remove('temp.html')
    return scripts

def _plotly_script():
    fig = go.Figure([go.Bar(
        x=['giraffes', 'orangutans', 'monkeys'], 
        y=[20, 14, 23]
    )])
    scripts = _get_scripts(fig)
    return f"{scripts[0]}\n{scripts[1]}"

def _extract_plotly_graph(plotly_fig):
    scripts = _get_scripts(plotly_fig)
    plotly_graph = scripts[2]
    graph_id = re.search(
        r"(?<=getElementById\(\")[a-zA-Z0-9-]*(?=\"\)\))", 
        plotly_graph
    ).group()
    plotly_graph = f'<div id="{graph_id}" class="plotly-graph-div"' \
                   + f' height="550"></div>\n{plotly_graph}'
    return plotly_graph

def _add_pandas(html_string, df):
    html_pandas = df.to_html().replace(
        '<table border="1" class="dataframe">',
        '<table class="table table-striped">'
    )
    html_string += f''' 
        {html_pandas}''' 
    return html_string

def _add_fig(html_string, fig):
    html_string += f'''
        {_extract_plotly_graph(fig)}'''
    return html_string

def _add_title(html_string, title):
    html_string += f'''
        <h1>{title}</h1>'''
    return html_string

def _add_subtitle(html_string, subtitle):
    html_string += f'''
        <h2>{subtitle}</h2>'''
    return html_string

def _add_subsubtitle(html_string, subsubtitle):
    html_string += f'''
        <h3>{subsubtitle}</h3>'''
    return html_string

def _add_text(html_string, text):
    html_string += f'''
        <p>{text}</p>'''
    return html_string

def generate_report(path, contents):
    add = {
        'pandas': _add_pandas, 'fig': _add_fig,
        'title': _add_title, 'subtitle': _add_subtitle,
        'subsubtitle': _add_subsubtitle,
        'text': _add_text,
    }
    
    html_string = f'''
<html>
    <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
        <style>body{{ margin:0 100; background:whitesmoke; }}</style>
    </head>
    <body>
        {_plotly_script()}'''
    
    for content_type, content in contents:
        html_string = add[content_type](html_string, content)
    
    html_string += '''
    </body>
</html>'''

    f = open(path, 'w')
    f.write(html_string)
    f.close()
    return 'Report generated'