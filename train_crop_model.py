import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset (assuming Crop_recommendation.csv from Kaggle)
# Download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
data_path = 'data/Crop_recommendation.csv'
if not os.path.exists(data_path):
    print("Please download Crop_recommendation.csv from Kaggle and place in data/ folder.")
    exit()

df = pd.read_csv(data_path)

# Features and target
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target = 'label'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, 'models/crop_model.pkl')
print("Model saved to models/crop_model.pkl")