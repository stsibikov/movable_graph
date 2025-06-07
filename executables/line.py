'''
Barebones prototype

Editing data is done with movable line

Limitations
---
The line can be moved horizontally without any restrictions -
users may lose it. And its a little ugly
'''
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd

x = np.linspace(0, 10, 100)
target = np.sin(x)
df = pd.DataFrame({'x': x, 'target': target})
df['pred_init'] = df['target'].shift(20)
df['pred_after_corr'] = df['pred_init']

app = dash.Dash(__name__)

initial_line_y = 0

app.layout = html.Div([
    dcc.Store(id='line-y-store', data=initial_line_y),
    dcc.Graph(
        id='graph',
        config={'editable': True}
    ),
    html.Div(id='line-position', style={'marginTop': 20})
])

def create_figure(line_y):
    df['pred_after_corr'] = df['pred_init'] + line_y
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df['x'], y=df['target'], mode='lines', name='target', marker=dict(color='purple'))
    )

    fig.add_trace(
        go.Scatter(x=df['x'], y=df['pred_init'], mode='lines', name='pred_init', marker=dict(color='green'), line=dict(dash='dot'))
    )

    fig.add_trace(
        go.Scatter(x=df['x'], y=df['pred_after_corr'], mode='lines', name='pred_after_corr', marker=dict(color='blue'))
    )

    fig.update_layout(
        shapes=[
            dict(
                type='line',
                x0=df['x'].min(),
                x1=df['x'].max(),
                y0=line_y,
                y1=line_y,
                line=dict(color='red', width=3),
                editable=True
            ),
            # the share will be editable bc of global config
            # # Static blue line (not editable)
            # dict(
            #     type='line',
            #     x0=df['x'].min(),
            #     x1=df['x'].max(),
            #     y0=static_line_y,
            #     y1=static_line_y,
            #     line=dict(color='blue', width=3, dash='dash'),
            #     editable=False
            # )
        ],
        # dragmode='drawline',
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(range=[df['x'].min(), df['x'].max()]),
        yaxis=dict(range=[-2, 2]),
        uirevision='fixed-ui',
    )
    return fig

@app.callback(
    Output('graph', 'figure'),
    Output('line-position', 'children'),
    Output('line-y-store', 'data'),
    Input('graph', 'relayoutData'),
    State('line-y-store', 'data')
)
def update_line(relayout_data, stored_line_y):
    line_y = stored_line_y if stored_line_y is not None else initial_line_y

    if relayout_data:
        for key, value in relayout_data.items():
            if key.startswith('shapes[0].y0'):
                line_y = value
            if key.startswith('shapes[0].y1'):
                line_y = value

    fig = create_figure(line_y)
    return fig, f'Line Y-position: {line_y:.3f}', line_y

if __name__ == '__main__':
    app.run(debug=True)
