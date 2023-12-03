from flask import Flask, request, jsonify , Response
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import json

# Create the app object
app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model1with53220')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Define a post method for our API.
@app.route('/')
def home():
    return "Welcome to the API"

# Define a post method for our API.
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON object containing the data sent by the client
    data = request.get_json()

    # Convert the list of dictionaries to a pandas DataFrame
    data_df = pd.DataFrame(data)

    # Convert the DataFrame to a numpy array
    data_np = data_df.to_numpy()

    # Convert the data to scale it using the trained scaler
    data_np = scaler.transform(data_np)

    # # Convert the data to float32
    data_np = data_np.astype(np.float32)

    # Reshape the data to match the input shape that the model expects
    data_np = data_np.reshape(1, data_np.shape[0], data_np.shape[1])

    # Make a prediction using the model
    prediction = model(data_np)

    # Inverse transform the prediction
    prediction = scaler.inverse_transform(prediction)    

    # Get the first prediction from the array of predictions
    output = prediction[0]

    # Convert the EagerTensor to a numpy array and then to a list
    output_list = output.tolist()

    ff = ['Close', 'Open', 'High', 'Low', 'Camarilla_R1', 'Camarilla_R2',
       'Camarilla_R3', 'Camarilla_R4', 'Camarilla_S1', 'Camarilla_S2',
       'Camarilla_S3', 'Camarilla_S4', 'ClassicR1', 'ClassicR2', 'ClassicR3',
       'ClassicR4', 'ClassicS1', 'ClassicS2', 'ClassicS3', 'ClassicS4',
       'DeMark_R1', 'DeMark_R2', 'DeMark_R3', 'DeMark_R4', 'DeMark_S1',
       'DeMark_S2', 'DeMark_S3', 'DeMark_S4', 'DeMark_X', 'Fibonacci_R1',
       'Fibonacci_R2', 'Fibonacci_R3', 'Fibonacci_R4', 'Fibonacci_S1',
       'Fibonacci_S2', 'Fibonacci_S3', 'Fibonacci_S4', 'Floor_R1', 'Floor_R2',
       'Floor_R3', 'Floor_R4', 'Floor_S1', 'Floor_S2', 'Floor_S3', 'Floor_S4',
       'Pivot_Points', 'Woodie_Pivot_Points', 'Woodie_R1', 'Woodie_R2',
       'Woodie_R3', 'Woodie_R4', 'Woodie_S1', 'Woodie_S2', 'Woodie_S3',
       'Woodie_S4', 'acc_dist_index', 'bb_hband', 'bb_lband', 'bb_mavg',
       'donchian_hband', 'donchian_lband_', 'donchian_mband_', 'ema',
       'fibonacci_0.236', 'fibonacci_0.382', 'fibonacci_0.5',
       'fibonacci_0.618', 'fibonacci_0.786', 'fibonacci_1', 'ichimoku_a',
       'ichimoku_b', 'ichimoku_base_line', 'ichimoku_conversion_line',
       'negative_volume_index', 'on_balance_volume', 'sma', 'supertrend',
       'vwap', 'wma']

    # Define the feature names
    feature_names = ff

    # Pair each feature name with its corresponding value
    output_dict = dict(zip(feature_names, output_list))

    # Convert the dictionary to a JSON string without sorting the keys
    output_json = json.dumps(output_dict, sort_keys=False)

    # Return the JSON string as a response
    return Response(output_json, mimetype='application/json')

    # Run the app
if __name__ == '__main__':
    app.run()
    #flask_app.run(host='0.0.0.0', port=8000)
