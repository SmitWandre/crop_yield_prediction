from flask import Flask, request, render_template, redirect, url_for
import joblib
import os
import numpy as np
import plotly.graph_objs as go
import plotly
import json

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'crop_yield_predictor.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    graphJSON = None
    if request.method == 'POST':
        try:
            temperature = float(request.form['temperature'])
            rainfall = float(request.form['rainfall'])
            soil_quality = float(request.form['soil_quality'])
            input_data = np.array([[temperature, rainfall, soil_quality]])
            prediction = model.predict(input_data)[0]

            temp_range = np.linspace(temperature - 5, temperature + 5, 50)
            yields = []
            for temp in temp_range:
                inp = np.array([[temp, rainfall, soil_quality]])
                yields.append(model.predict(inp)[0])
                
            trace = go.Scatter(
                x = temp_range,
                y = yields,
                mode = 'lines+markers',
                name = 'Predicted Yield'
            )
            layout = go.Layout(
                title = 'Predicted Crop Yield vs Temperature',
                xaxis = dict(title='Temperature (°C)'),
                yaxis = dict(title='Crop Yield')
            )
            fig = go.Figure(data=[trace], layout=layout)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction, graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)

