import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Split the data into features and labels
X = data.iloc[:, :-1]  # Features: N, P, K, temperature, humidity, ph, rainfall
y = data.iloc[:, -1]   # Label: crop name

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model using pickle
pickle.dump(model, open("model.pkl", "wb"))

# =========================
# 🔮 Predict for new input
# =========================

# Define your new feature input
new_features = [[117, 32, 34, 26.27, 52.12, 6.75, 127.17]]  # Example input

# Wrap into a DataFrame to match training input structure
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
new_df = pd.DataFrame(new_features, columns=feature_names)

# Load model and make prediction
loaded_model = pickle.load(open("model.pkl", "rb"))
predicted_crop = loaded_model.predict(new_df)

print("🌾 Predicted Crop:", predicted_crop[0])
