'''
A proper prototype

Editing data is done with input boxes adding and
subtracting from changable column values based on
values of the base, unchangable column.

Features
---
Choose to move entire line

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
from threading import Timer
from pathlib import Path
from datetime import date

from movable_graph import to_list

host = '127.0.0.1'
port = 8050

project_dir = Path(__file__).resolve().parent.parent

display_cols = ['target', 'pred_init', 'pred']


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
weekdays = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
]

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='store-data', data=df.to_json(orient='split')),
    dcc.Store(id='selected-base-indexes', data=[]),
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
    html.Br(), html.Br(),
    html.Label("Move entire line"),
    dcc.Input(id='change-mult-all', type='number', value=.0, step=0.1, debounce=True),
    html.Br(), html.Br(),
    html.Label("Move line on specific weekdays"),
    dcc.Dropdown(
        id='weekday-dropdown',
        options=[{'label': wd, 'value': wd} for wd in weekdays],
        multi=True,
        placeholder="Select day(s) of week"
    ),
    dcc.Input(id='change-mult-weekdays', type='number', value=.0, step=0.1, debounce=True),
    html.Br(), html.Br(),
    html.Label("Move values on selected date"),
    dcc.DatePickerSingle(
        id='date-picker-single',
        initial_visible_month=date.today(),
        date=date.today(),
        display_format='DD.MM.YY'
    ),
    dcc.Input(id='change-mult-date', type='number', value=.0, step=0.1, debounce=True),
    html.Br(), html.Br(),
    dcc.Graph(id='line-plot'),
    html.Button("Export CSV", id="export-csv-btn"),
])

@app.callback(
    Output('line-plot', 'figure'),
    Output('selected-base-indexes', 'data'),
    Output('change-mult-all', 'value'),
    Output('change-mult-weekdays', 'value'),
    Output('change-mult-date', 'value'),
    Input('obj-id-dropdown', 'value'),
    Input('segment-dropdown', 'value'),
    State('store-data', 'data'),
)
def update_plot_and_selected_indexes(
    selected_objs,
    selected_segments,
    df
):
    '''
    Updates the unmodified parts of the plot, when user changes ids or segments

    When switching, all modifiers are reset to zero
    '''
    selected_objs = to_list(selected_objs)
    selected_segments = to_list(selected_segments)

    logger.debug(
        'update_plot_and_selected_indexes:'
        f'\n\t{dash.callback_context.triggered = }'
        f'\n\t{selected_objs = }'
        f'\n\t{selected_segments = }'
    )

    # dataframe with current filters
    df: pd.DataFrame = pd.read_json(io.StringIO(df), orient='split', convert_dates=['dt'])

    if selected_objs:
        df = df[df['obj_id'].isin(selected_objs)]
    if selected_segments:
        df = df[df['segment'].isin(selected_segments)]

    # for updating df in `update_modifiable_col`
    selected_base_indexes = df.index

    df = df.groupby(['dt', 'weekday', 'week'], as_index=False)[display_cols].sum()

    traces = []

    for display_col in display_cols:
        traces.append(
            go.Scatter(
                x=df['dt'], y=df[display_col], mode='lines', name=display_col,
                # hovertemplate= (
                #     'Date: %{x}<br>'
                #     f'{display_col}:'
                #     '%{y}<br>'
                #     'Weekday: %{customdata[0]}<br>'
                #     'Week: %{customdata[1]}<extra></extra>',
                # ),
                # customdata=df[['weekday', 'week']].values
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(title="Graph", xaxis_title='dt', yaxis_title='Value')

    return fig, selected_base_indexes, .0, .0, .0

@app.callback(
    Output('store-data', 'data'),
    Output('line-plot', 'figure', allow_duplicate=True),
    Output('change-mult-all', 'value', allow_duplicate=True),
    Output('change-mult-weekdays', 'value', allow_duplicate=True),
    Output('change-mult-date', 'value', allow_duplicate=True),
    Input('change-mult-all', 'value'),
    Input('weekday-dropdown', 'value'),
    Input('change-mult-weekdays', 'value'),
    Input('date-picker-single', 'date'),
    Input('change-mult-date', 'value'),
    State('line-plot', 'figure'),
    State('selected-base-indexes', 'data'),
    State('store-data', 'data'),
    prevent_initial_call=True,
)
def update_modifiable_col(
    change_mult_all,
    chosen_weekdays,
    change_mult_weekdays,
    chosen_date,
    change_mult_date,
    fig,
    selected_base_indexes,
    df,
):
    '''
    Applies multipliers to displayed and stored data
    '''
    if not change_mult_all and not (chosen_weekdays and change_mult_weekdays) and not (chosen_date and change_mult_date):
        return dash.no_update

    df = pd.read_json(io.StringIO(df), orient='split', convert_dates=['dt'])

    # no need to filter again - weve done that
    # dfs is for display, df is for storing updated dataframe
    dfs = df.loc[selected_base_indexes, :]

    logger.debug(
        'update_modifiable_col'
        f'\n\t{dash.callback_context.triggered = }'
        f'\n\t{change_mult_all = }'
        f'\n\t{chosen_weekdays = }'
        f'\n\t{change_mult_weekdays = }'
        f'\n\t{chosen_date = }'
        f'\n\t{change_mult_date = }'
    )

    dfs: pd.DataFrame = dfs.groupby(['dt', 'weekday', 'week'], as_index=False)[['pred_init', 'pred']].sum()

    if change_mult_all:
        dfs['pred'] += dfs['pred_init'] * change_mult_all
        df.loc[selected_base_indexes, 'pred'] += df.loc[selected_base_indexes, 'pred_init'] * change_mult_all

    if chosen_weekdays and change_mult_weekdays:
        mask = dfs['weekday'].isin(chosen_weekdays)
        dfs.loc[
            mask, 'pred'
        ] += dfs.loc[
            mask, 'pred_init'
        ] * change_mult_weekdays

        mask = df.index.isin(selected_base_indexes) & df['weekday'].isin(chosen_weekdays)
        df.loc[
            mask, 'pred'
        ] += df.loc[
            mask, 'pred_init'
        ] * change_mult_weekdays

    if chosen_date and change_mult_date:
        mask = dfs['dt'].dt.normalize() == chosen_date
        dfs.loc[
            mask, 'pred'
        ] += dfs.loc[
            mask, 'pred_init'
        ] * change_mult_date

        mask = df.index.isin(selected_base_indexes) & (df['dt'].dt.normalize() == chosen_date)
        df.loc[
            mask, 'pred'
        ] += df.loc[
            mask, 'pred_init'
        ] * change_mult_date

    # update only the modified graph trace
    for trace in fig['data']:
        if trace['name'] == 'pred':
            # Update the trace data as needed
            trace['y'] = dfs['pred']
            break

    # save updated dataframe to reflect changes when the user switches back
    # or downloads the results
    df = df.to_json(orient='split')

    return df, fig, .0, .0, .0


@app.callback(
    Output("export-csv-btn", "children"),
    Input("export-csv-btn", "n_clicks"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def save_csv_to_disk(n_clicks, df):
    if n_clicks:
        df = pd.read_json(io.StringIO(df), orient='split', convert_dates=['dt'])
        file_path = project_dir / 'data' / 'panel_input_output.csv'
        df.to_csv(file_path, index=False)
        return 'Results saved'
    return 'Export results'


if __name__ == '__main__':
    # Timer(3, webbrowser.open_new(f"http://{host}:{port}"))
    app.run(
        host=host,
        port=port,
        debug=True
    )
