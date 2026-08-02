from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Enable CORS so your Vite frontend can communicate locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved artifacts
artifacts = joblib.load("ml_pipeline/models/model.pkl")
model = artifacts["model"]
scaler = artifacts["scaler"]

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: PredictionRequest):
    # Ensure input has 20 features
    if len(data.features) != 20:
        return {"error": "Exactly 20 features are required."}
    
    X_input = np.array([data.features])
    X_scaled = scaler.transform(X_input)
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])
    
    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }
