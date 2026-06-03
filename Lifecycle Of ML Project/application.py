from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## impot ridge regressor and standard scaler pickle files
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST']) 
def predict_data():
    if request.method == 'POST':
        # Get the input values from the form
        temperature = float(request.form['temperature'])
        rh = float(request.form['rh'])
        ws = float(request.form['ws'])
        rain = float(request.form['rain'])
        ffmc = float(request.form['ffmc'])
        dmc = float(request.form['dmc'])
        isi = float(request.form['isi'])
        classes = float(request.form['classes'])
        region = float(request.form['region'])

        # Create a numpy array with the input values
        input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])

        # Scale the input data
        scaled_data = standard_scaler.transform(input_data)

        # Make the prediction
        prediction = ridge_model.predict(scaled_data)

        # Return the prediction result
        return render_template('home.html', prediction=prediction[0])
    else:
        return render_template('home.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'API is running'
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)