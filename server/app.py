from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as ex
import json
import argparse
import flask
import numpy as np

sensor_list = ''
importance = ''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirQual server")
    parser.add_argument(
        '-d', '--debug', help='Turn on debug mode', action='store_true')
    parser.add_argument('sensor_list', help='list of sensors (json file)')
    parser.add_argument(
        'importance', help='sensor importance data (json file)')
    args = parser.parse_args()

    sensor_list = args.sensor_list
    importance = args.importance
else:
    sensor_list = 'sensors_list.json'
    importance = 'importance.json'

# Constants
img_width = 1030
img_height = 960
scale_factor = 1

np.random.seed(1)

with open(sensor_list, 'r') as f:
    sensors = json.load(f)

sensors_len = len(sensors)
sensors_ids = list(map(lambda x: x['id'], sensors))

with open(importance, 'r') as f:
    importance = json.load(f)

inner_ids = list(importance.keys())
outer_ids = list(importance[inner_ids[0]].keys())

importance_matrix = [[0 for _ in range(sensors_len)]
                     for _ in range(sensors_len)]

for inner_id in inner_ids:
    matrix_inner_id = 0
    inner_dict = importance[inner_id]

importance_dict = {}
for sensor_id in sensors_ids:
    if str(sensor_id) in inner_ids:
        importance_array_inner = []
        for sensor_id_inner in sensors_ids:
            if str(sensor_id_inner) in outer_ids:
                importance_array_inner.append(
                    importance[str(sensor_id)][str(sensor_id_inner)])
            else:
                importance_array_inner.append(0.002)
        importance_dict[sensor_id] = importance_array_inner
    else:
        importance_array_inner = []
        for sensor_id_inner in sensors_ids:
            if str(sensor_id_inner) in outer_ids:
                importance_array_inner.append(0.005)
            else:
                importance_array_inner.append(0.002)
        importance_dict[sensor_id] = importance_array_inner

x = list(map(lambda s: s['location']['longitude'], sensors))
y = list(map(lambda s: s['location']['latitude'], sensors))

f = go.FigureWidget([go.Scatter(x=x, y=y, mode='markers')])

# Configure axes
f.update_xaxes(
    visible=True,
    range=[19.70, 20.20],  # [0, img_width * scale_factor]
)

f.update_yaxes(
    visible=True,
    range=[49.90, 50.20],  # [0, img_height * scale_factor],
    # the scaleanchor attribute ensures that the aspect ratio stays constant
    # scaleanchor='x'
)

# Add image
f.add_layout_image(
    dict(
        x=19.70,
        sizex=0.5,
        y=50.20,
        sizey=0.3,
        xref='x',
        yref='y',
        opacity=0.5,
        layer='below',
        sizing='stretch',
        source='https://raw.githubusercontent.com/PKopel/AirQual/main/charts/cracow3.png')
)

# Configure other layout
f.update_layout(
    width=img_width * scale_factor,
    height=img_height * scale_factor,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
)

scatter = f.data[0]
colors = np.random.rand(len(x))
scatter.marker.color = colors
scatter.marker.showscale = True
scatter.marker.size = [10] * len(x)
f.layout.hovermode = 'closest'

server = flask.Flask(__name__)

app = Dash(__name__, server=server)

app.layout = html.Div([
    html.H4('Interactive map of sensor importance'),
    dcc.Graph(id='sensor-importance', figure=f)
])


@app.callback(
    Output('sensor-importance', 'figure'),
    Input('sensor-importance', 'clickData')
)
def update_points(clickData):
    if clickData:
        i = clickData['points'][0]['pointIndex']
        c = importance_dict[sensors_ids[i]]
        with f.batch_update():
            scatter.marker.color = c
            scatter.marker.size = list(map(lambda x: x * 5000, c))
    return f


if __name__ == "__main__":
    app.run_server(debug=args.debug)
else:
    server = app.server
