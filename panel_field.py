import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import io
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create a StreamHandler that outputs to stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)

logger.addHandler(stdout_handler)

np.random.seed(0)
objects = ['Object A', 'Object B', 'Object C']
n_points = 50

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
    dcc.Store(id='store-data', data={
        'df': df.to_json(orient='split'),
        # Removed multipliers storage
    }),
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
    dcc.Graph(id='line-plot'),
    html.Button("Export CSV", id="export-csv-btn")
])

@app.callback(
    Output('store-data', 'data'),
    Output('multiplier-all', 'value'),
    Output('multiplier-every-second', 'value'),
    Output('line-plot', 'figure'),
    Input('object-dropdown', 'value'),
    Input('multiplier-all', 'value'),
    Input('multiplier-every-second', 'value'),
    State('store-data', 'data')
)
def update_data_and_plot(selected_obj, mult_all, mult_second, store_data):
    df = pd.read_json(io.StringIO(store_data['df']), orient='split')

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Reset multipliers inputs to 1.0 when object changes
    if trigger_id == 'object-dropdown':
        mult_all = 1.0
        mult_second = 1.0

    logger.debug(
        f'{ctx.triggered = }'
        f'\n{selected_obj = }'
        f'\n{mult_all = }'
        f'\n{mult_second = }\n'
    )

    # Filter data for selected object
    dff = df[df['object'] == selected_obj].copy()

    # Apply multipliers directly to pred_init to update pred
    dff['pred'] = dff['pred_init'] * (mult_all if mult_all is not None else 1.0)
    dff.loc[dff.index[1::2], 'pred'] *= (mult_second if mult_second is not None else 1.0)

    # Update global dataframe pred values for this object
    df.loc[dff.index, 'pred'] = dff['pred']

    # Save updated dataframe back to store
    store_data['df'] = df.to_json(orient='split')

    # Create plot traces
    trace_target = go.Scatter(x=dff['x'], y=dff['target'], mode='lines', name='target')
    trace_pred_init = go.Scatter(x=dff['x'], y=dff['pred_init'], mode='lines', name='pred_init')
    trace_pred = go.Scatter(x=dff['x'], y=dff['pred'], mode='lines', name='pred')

    fig = go.Figure(data=[trace_target, trace_pred_init, trace_pred])
    fig.update_layout(title=f"Lines for {selected_obj}", xaxis_title='x', yaxis_title='Value')

    return store_data, mult_all, mult_second, fig

@app.callback(
    Output("export-csv-btn", "children"),
    Input("export-csv-btn", "n_clicks"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def save_csv_to_disk(n_clicks, store_data):
    if n_clicks:
        df = pd.read_json(io.StringIO(store_data['df']), orient='split')
        filename = "df.csv"
        df.to_csv(filename, index=False)
        return f"Saved to {filename}"
    return "Export CSV"

if __name__ == '__main__':
    app.run(debug=True)
