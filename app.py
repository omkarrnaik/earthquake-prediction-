
# import numpy as np
# from flask import Flask, request,render_template
# import pickle

# # Create flask app
# flask_app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))

# @flask_app.route("/")
# def Home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods = ["POST"])
# def predict():
#     float_features = [float(x) for x in request.form.values()]
#     features = [np.array(float_features)]
#     prediction = model.predict(features)
#     return render_template("index.html", prediction_text = "The Predicted Crop is {}".format(prediction))

# if __name__ == "__main__":
#     flask_app.run(debug=True)

from flask import send_from_directory
import os
import numpy as np
from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/farm.jpg')
def send_image():
    return send_from_directory(os.getcwd(), 'farm.jpg')


# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract and convert input values from form
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Send prediction to the template
        return render_template("index.html", prediction_text=f"The Predicted Crop is {prediction}")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

