from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import joblib
import os

# ---------------- CONFIG ---------------- #

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "gesture_model_xyz.pkl"

# ---------------- APP ---------------- #

app = FastAPI()

# Mount static folder (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ---------------- LOAD MODEL ---------------- #

print("🔄 Loading model...")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded!")
except Exception as e:
    print("❌ Model load failed:", e)
    model = None

# ---------------- REQUEST FORMAT ---------------- #

class LandmarkData(BaseModel):
    data: list  # XYZ flattened list (63 values)

# ---------------- ROUTES ---------------- #

@app.get("/")
def home():
    """Serve landing page"""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/predict")
def predict(landmarks: LandmarkData):
    """Predict gesture from XYZ landmarks"""

    if model is None:
        return {"error": "Model not loaded"}

    try:
        data = np.array(landmarks.data)

        # Ensure correct shape
        if data.shape[0] != 63:
            return {"error": f"Expected 63 values, got {data.shape[0]}"}

        data = data.reshape(1, -1)

        prediction = model.predict(data)[0]

        # Confidence
        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(data)) * 100
        else:
            confidence = 0.0

        return {
            "prediction": str(prediction),
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}

port = int(os.environ.get("PORT", 8501))
