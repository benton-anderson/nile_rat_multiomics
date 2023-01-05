"""
App Layout:

    Tab 1)
        Volcano plots for lipids and polar metabolites
            Should I combine into one plot, or two separate plots, or render each on-demand?
            I think one plot is clearer
                Dots can be filtered by Type == lipid or metabolite, or within superclass, or within lipid class
    Tab 2)
        Slope vs slope plots
            One plot
            Same dot filtering as Tab 1
    Tab 3)
        Quant vs OGTT plot
            The default plot is TG 20:5_22:6_22:6
            Drop down menu with all IDs, then all un-IDs listed by their "l_123" and mz_RT unique name

            Also include all p-values



"""


import dash
from dash import html
from dash import dcc
import graphs

app = dash.Dash()

graph1 = dcc.Graph(id='graph1', figure=graphs.test_figure)

app.layout = html.Div([
    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label="Tab 1", value="tab-1", children=[graph1]),
        dcc.Tab(label="Tab 2", value="tab-2", children=[
            dcc.Graph(id='graph-2', figure={
                'data': [{'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'NYC'}],
                'layout': {'title': 'Dash Data Visualization'}
            })
        ])
    ])
])

if __name__ == '__main__':
    print('SERVER RUNNING')
    app.run_server(debug=True)





















































