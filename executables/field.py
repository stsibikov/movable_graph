import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Sample initial data
np.random.seed(0)
x = np.arange(20)
target = np.sin(x / 3)
pred_init = target + np.random.normal(0, 0.1, size=len(x))
pred = pred_init.copy()

# Create initial DataFrame
df = pd.DataFrame({
    'x': x,
    'target': target,
    'pred_init': pred_init,
    'pred': pred
})

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Input(
        id='input-multiplier-all',
        type='number',
        placeholder='Multiplier for all pred points',
        debounce=True,
        style={'marginRight': '10px'}
    ),
    dcc.Input(
        id='input-multiplier-every-second',
        type='number',
        placeholder='Multiplier for every second pred point',
        debounce=True
    ),
    dcc.Graph(id='line-graph'),
    # Store the dataframe as JSON in dcc.Store for sharing between callbacks
    dcc.Store(id='df-store', data=df.to_json(date_format='iso', orient='split'))
])


@app.callback(
    Output('df-store', 'data'),
    Input('input-multiplier-all', 'value'),
    Input('input-multiplier-every-second', 'value'),
    State('df-store', 'data')
)
def update_dataframe(mult_all, mult_second, jsonified_df):
    # Load DataFrame from stored JSON
    df = pd.read_json(jsonified_df, orient='split')

    # Start from pred_init to avoid compounding multiplications
    pred = df['pred_init'].copy()

    if mult_all is not None:
        pred = pred * mult_all

    if mult_second is not None:
        # Multiply every second point (index 1,3,5...) by mult_second
        pred.iloc[1::2] = pred.iloc[1::2] * mult_second

    # Update pred column in DataFrame
    df['pred'] = pred

    # Return updated DataFrame as JSON
    return df.to_json(date_format='iso', orient='split')


@app.callback(
    Output('line-graph', 'figure'),
    Input('df-store', 'data')
)
def update_graph(jsonified_df):
    df = pd.read_json(jsonified_df, orient='split')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['x'], y=df['target'], mode='lines', name='target'))
    fig.add_trace(go.Scatter(x=df['x'], y=df['pred_init'], mode='lines', name='pred_init'))
    fig.add_trace(go.Scatter(x=df['x'], y=df['pred'], mode='lines', name='pred'))

    fig.update_layout(title='Target, Pred_init and Pred Lines',
                      xaxis_title='X',
                      yaxis_title='Value')

    return fig


if __name__ == '__main__':
    app.run(debug=True)
