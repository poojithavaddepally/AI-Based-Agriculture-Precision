from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = FastAPI(title="Agri Platform ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
crop_model = None
disease_model = None
class_indices = None

@app.on_event("startup")
async def load_models():
    global crop_model, disease_model, class_indices
    try:
        crop_model = joblib.load('models/crop_model.pkl')
        disease_model = tf.keras.models.load_model('models/disease_model.h5')
        with open('models/class_indices.json', 'r') as f:
            class_indices = json.load(f)
    except Exception as e:
        print(f"Error loading models: {e}")

class CropPredictionRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class CropPrediction(BaseModel):
    crop: str
    confidence: float

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float
    remedy: str

def get_remedy(disease: str) -> str:
    remedies = {
        "Tomato___Bacterial_spot": "Apply copper-based fungicide and ensure proper spacing.",
        "Tomato___Early_blight": "Remove affected leaves and apply fungicide.",
        "Tomato___Late_blight": "Isolate plant and use systemic fungicide.",
        "Tomato___Leaf_Mold": "Improve ventilation and apply fungicide.",
        "Tomato___Septoria_leaf_spot": "Remove infected leaves and apply fungicide.",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Use insecticidal soap.",
        "Tomato___Target_Spot": "Apply fungicide and avoid overhead watering.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants and control whiteflies.",
        "Tomato___Tomato_mosaic_virus": "Remove infected plants and disinfect tools.",
        "Tomato___healthy": "No action needed."
    }
    return remedies.get(disease, "Inspect the affected leaves carefully, remove damaged tissue, improve airflow, and avoid overhead watering. Use a suitable disease control measure and monitor the plant closely; consult your local extension service if the issue persists.")

@app.post("/predict-crop", response_model=List[CropPrediction])
async def predict_crop(request: CropPredictionRequest):
    if crop_model is None:
        raise HTTPException(status_code=500, detail="Crop model not loaded")

    features = np.array([[request.N, request.P, request.K, request.temperature, request.humidity, request.ph, request.rainfall]])
    probabilities = crop_model.predict_proba(features)[0]
    classes = crop_model.classes_

    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]
    predictions = [
        CropPrediction(crop=classes[i], confidence=round(probabilities[i] * 100, 2))
        for i in top_indices
    ]

    return predictions

@app.post("/predict-disease", response_model=DiseasePrediction)
async def predict_disease(file: UploadFile = File(...)):
    if disease_model is None or class_indices is None:
        raise HTTPException(status_code=500, detail="Disease model not loaded")

    # Read and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = disease_model.predict(image_array)[0]
    predicted_class_index = np.argmax(predictions)
    confidence = round(predictions[predicted_class_index] * 100, 2)

    # Get class name
    class_names = {v: k for k, v in class_indices.items()}
    disease = class_names[predicted_class_index]

    remedy = get_remedy(disease)

    return DiseasePrediction(disease=disease, confidence=confidence, remedy=remedy)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
