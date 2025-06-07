import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

app = dash.Dash(__name__)

fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 10], y=[1, 3], mode='lines+markers'))
fig.update_layout(
    shapes=[dict(type='line', x0=0, x1=10, y0=2, y1=2, line=dict(color='red', width=3), editable=True)],
    yaxis=dict(range=[0, 5]),
    xaxis=dict(range=[0, 10])
)

app.layout = html.Div([
    dcc.Graph(id='graph', figure=fig, config={'editable': True, 'edits': {'shapePosition': True}}),
    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    Input('graph', 'relayoutData')
)
def display_relayout_data(relayout_data):
    if relayout_data and any(key.startswith('shapes') for key in relayout_data):
        return str(relayout_data)
    return "Drag the red line to move it."

if __name__ == '__main__':
    app.run(debug=True)
