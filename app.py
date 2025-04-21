# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import pickle

# # Initialize the app
# app = Flask(__name__)

# # Load the trained model
# model = pickle.load(open('model.pkl', 'rb'))

# # # Label map
# # label_map = {
# #     0: "No Earthquake",
# #     1: "Mild Earthquake",
# #     2: "Severe Earthquake"
# # }
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     lat = data['lat']
#     lon = data['lon']
#     depth = data['depth']

#     # Convert to numpy array and predict
#     features = np.array([[lat, lon, depth]], dtype='float32')
#     prediction = model.predict(features)

#     # If your model outputs a float like 5.3
#     magnitude = round(float(prediction[0]), 2)

#     return jsonify({'prediction': f"{magnitude} magnitude"})


# # Route to serve the HTML page
# @app.route('/')
# def home():
#     return render_template('index.html')  # Make sure your HTML file is named 'index.html' and inside a 'templates' folder

# # API route to handle predictions
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     lat = data['lat']
#     lon = data['lon']
#     depth = data['depth']

#     # Convert to numpy array and predict
#     features = np.array([[lat, lon, depth]], dtype='float32')
#     prediction = model.predict(features)

#     label = label_map[int(prediction[0])]
#     return jsonify({'prediction': label})

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Initialize the app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' is inside a 'templates' folder

# API route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    lat = data['lat']
    lon = data['lon']
    depth = data['depth']

    # Convert to numpy array and predict
    features = np.array([[lat, lon, depth]], dtype='float32')
    prediction = model.predict(features)

    # Assuming model outputs a float value (e.g., 6.41)
    magnitude = round(float(prediction[0]), 2)

    return jsonify({'prediction': f"{magnitude} magnitude"})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

