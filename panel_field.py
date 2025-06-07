import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Sample data creation
np.random.seed(0)
objects = ['Object A', 'Object B', 'Object C']
n_points = 50

# Create a dataframe with columns: object, x, target, pred_init, pred
df_list = []
for obj in objects:
    x = np.arange(n_points)
    target = np.sin(x / 5) + np.random.normal(0, 0.1, n_points)
    pred_init = target + np.random.normal(0, 0.2, n_points)
    pred = pred_init.copy()
    df_list.append(pd.DataFrame({
        'object': obj,
        'x': x,
        'target': target,
        'pred_init': pred_init,
        'pred': pred
    }))
df = pd.concat(df_list, ignore_index=True)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Multi-line Plot with Object Selection and Float Inputs"),
    dcc.Dropdown(
        id='object-dropdown',
        options=[{'label': obj, 'value': obj} for obj in objects],
        value=objects[0],
        clearable=False
    ),
    html.Br(),
    html.Label("Multiply entire 'pred' line by:"),
    dcc.Input(id='multiplier-all', type='number', value=1.0, step=0.1),
    html.Br(), html.Br(),
    html.Label("Multiply every second point of 'pred' line by:"),
    dcc.Input(id='multiplier-every-second', type='number', value=1.0, step=0.1),
    html.Br(), html.Br(),
    dcc.Graph(id='line-plot')
])

@app.callback(
    Output('line-plot', 'figure'),
    Output('multiplier-all', 'value'),
    Output('multiplier-every-second', 'value'),
    Input('object-dropdown', 'value'),
    Input('multiplier-all', 'value'),
    Input('multiplier-every-second', 'value')
)
def update_plot(selected_obj, mult_all, mult_second):
    global df

    # Reset multipliers to 1 if the trigger was the dropdown change
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger_id = None
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'object-dropdown':
        mult_all = 1.0
        mult_second = 1.0

    # Filter data for selected object
    dff = df[df['object'] == selected_obj].copy()

    # Apply multipliers
    dff['pred'] = dff['pred_init'] * mult_all
    dff.loc[dff.index[1::2], 'pred'] *= mult_second

    # Update global dataframe
    df.loc[dff.index, 'pred'] = dff['pred']

    # Create traces
    trace_target = go.Scatter(x=dff['x'], y=dff['target'], mode='lines', name='target')
    trace_pred_init = go.Scatter(x=dff['x'], y=dff['pred_init'], mode='lines', name='pred_init')
    trace_pred = go.Scatter(x=dff['x'], y=dff['pred'], mode='lines', name='pred')

    fig = go.Figure(data=[trace_target, trace_pred_init, trace_pred])
    fig.update_layout(title=f"Lines for {selected_obj}", xaxis_title='x', yaxis_title='Value')

    return fig, mult_all, mult_second

app.run(debug=True)
