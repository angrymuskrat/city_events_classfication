import plotly.express as px

def plot_clusters(df, color='label', hover_name='title'):
    fig = px.scatter(df, x="x", y="y", color=color, hover_name=hover_name)
    fig.show()
    return

def wide_screen():
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:70% !important; }</style>"))
    return

def standart_screen():
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:70% !important; }</style>"))
    return