import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess data
data = pd.read_csv("dataset.csv")
data = np.array(data)
print(data)

X = data[:, 0:-1]
y = data[:, -1]
y = y.astype('int')
X = X.astype('float32')  # Keeping features as float if coordinates/depths are not integers

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Save the model
pickle.dump(rfc, open('model.pkl', 'wb'))

# ------------------------
# Sample prediction
# ------------------------

# Load the saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Sample input with 3 features (latitude, longitude, depth)
sample_input = np.array([[29.06, 77.42, 5.0]])
sample_input = sample_input.astype('float32')

# Make prediction
prediction = loaded_model.predict(sample_input)

# Map prediction to label
label_map = {
    0: "No Earthquake",
    1: "Mild Earthquake",
    2: "Severe Earthquake"
}

# Print human-readable result
print("Sample Prediction:", label_map[int(prediction[0])])





