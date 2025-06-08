'''
A proper prototype

Editing data is done with input boxes adding and
subtracting from changable column values based on
values of the base, unchangable column.

Features
---
User can choose data to display based on different criteria,
and the data changes will work

Limitations
---
'''
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import io
import logging
import sys
import webbrowser
from pathlib import Path
from threading import Timer
from datetime import date

host = '127.0.0.1'
port = 8050

project_dir = Path(__file__).resolve().parent.parent


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)

    logger.addHandler(stdout_handler)
    return logger


def get_data():
    path = project_dir / 'data' / 'dashboard_input.csv'

    df = pd.read_csv(path, parse_dates=['dt'])

    # `pred` will be modified, `pred_init` will stay the same
    # to act as a reference point
    df['pred_init'] = df['pred']

    df['weekday'] = df['dt'].dt.day_name()

    df['week'] = df['dt'].dt.isocalendar().week
    return df


logger = get_logger()
df = get_data()

objects = list(df['obj_id'].unique())
segments = list(df['segment'].unique())


app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='store-data', data={
        'df': df.to_json(orient='split')
    }),
    dcc.Dropdown(
        id='obj-id-dropdown',
        placeholder='Object ID(s)',
        options=[{'label': obj, 'value': obj} for obj in objects],
        value=objects[0],
        clearable=False,
        searchable=True,
        multi=True
    ),
    dcc.Dropdown(
        id='segment-dropdown',
        options=[{'label': s, 'value': s} for s in segments],
        multi=True,
        placeholder="Select Segment(s)"
    ),
    html.Br(),
    html.Label("Move entire line"),
    dcc.Input(id='change-mult-all', type='number', value=.0, step=0.1, debounce=True),
    html.Br(), html.Br(),
    html.Label("Multiply every second point of 'pred' line by:"),
    dcc.Input(id='change-mult-every-second', type='number', value=.0, step=0.1),
    html.Br(), html.Br(),
    dcc.DatePickerSingle(
        id='date-picker-single',
        initial_visible_month=date.today(),
        date=date.today(),
        display_format='DD.MM.YY'
    ),
    html.Label("Move values on selected date"),
    dcc.Input(id='change-mult-date', type='number', value=.0, step=0.1, debounce=True),
    html.Br(), html.Br(),
    dcc.Graph(id='line-plot'),
    html.Button("Export CSV", id="export-csv-btn"),
])

@app.callback(
    Output('store-data', 'data'),
    Output('change-mult-all', 'value'),
    Output('change-mult-every-second', 'value'),
    Output('line-plot', 'figure'),
    Input('obj-id-dropdown', 'value'),
    Input('segment-dropdown', 'value'),
    Input('change-mult-all', 'value'),
    Input('change-mult-every-second', 'value'),
    State('store-data', 'data'),
)
def update_data_and_plot(
    selected_objs,
    selected_segments,
    change_mult_all,
    change_mult_every_second,
    store_data
):
    callback_context = dash.callback_context
    trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else None

    if trigger_id == 'obj-id-dropdown':
        change_mult_all = .0
        change_mult_every_second = .0

    logger.debug(
        f'{callback_context.triggered = }'
        f'\n{selected_objs = }'
        f'\n{selected_segments = }'
        f'\n{change_mult_all = }'
        f'\n{change_mult_every_second = }\n'
    )

    df = pd.read_json(io.StringIO(store_data['df']), orient='split')

    dfs = df.copy()

    if selected_objs:
        dfs = dfs[dfs['obj_id'].isin(selected_objs)]
    if selected_segments:
        dfs = dfs[dfs['segment'].isin(selected_segments)]

    # for updating df later
    affected_indexes = dfs.index

    dfs = dfs.groupby('dt', as_index=False)[['target', 'pred_init', 'pred']].sum()

    dfs['pred'] += dfs['pred_init'] * change_mult_all
    dfs.loc[dfs.index[1::2], 'pred'] += dfs.loc[dfs.index[1::2], 'pred_init'] * change_mult_every_second

    trace_target = go.Scatter(x=dfs['dt'], y=dfs['target'], mode='lines', name='target')
    trace_pred_init = go.Scatter(x=dfs['dt'], y=dfs['pred_init'], mode='lines', name='pred_init')
    trace_pred = go.Scatter(x=dfs['dt'], y=dfs['pred'], mode='lines', name='pred')

    fig = go.Figure(data=[trace_target, trace_pred_init, trace_pred])
    fig.update_layout(title=f"Lines for {selected_objs}", xaxis_title='dt', yaxis_title='Value')

    # apply changes to original df according to selection
    df.loc[affected_indexes, 'pred'] += df['pred_init'] * change_mult_all
    df.loc[affected_indexes[1::2], 'pred'] += df.loc[affected_indexes[1::2], 'pred_init'] * change_mult_every_second

    # save updated dataframe to reflect changes
    store_data['df'] = df.to_json(orient='split')

    return store_data, .0, .0, fig

@app.callback(
    Output("export-csv-btn", "children"),
    Input("export-csv-btn", "n_clicks"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def save_csv_to_disk(n_clicks, store_data):
    if n_clicks:
        df = pd.read_json(io.StringIO(store_data['df']), orient='split')
        file_path = project_dir / 'data' / 'panel_input_output.csv'
        df.to_csv(file_path, index=False)
        return 'Results saved'
    return 'Export results'


if __name__ == '__main__':
    Timer(3, webbrowser.open_new(f"http://{host}:{port}"))
    app.run(
        host=host,
        port=port,
        debug=True
    )
