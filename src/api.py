import joblib
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import os

# Load model
model = joblib.load("models/best_model.pkl")

# Create app
app = FastAPI()

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# Input schema with validation
class InputData(BaseModel):
    content_length_min: float = Field(..., ge=10, le=400)
    user_tenure_months: int = Field(..., ge=1, le=200)
    is_premium: int = Field(..., ge=0, le=1)
    genre_score: int = Field(..., ge=1, le=10)

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "best_model",
        "version": "1.0"
    }

# Prediction endpoint
@app.post("/score")
def score(data: InputData):
    try:
        features = [[
            data.content_length_min,
            data.user_tenure_months,
            data.is_premium,
            data.genre_score
        ]]

        prediction = model.predict(features)[0]

        # Logging (Task 4 prep)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": data.dict(),
            "prediction": float(prediction)
        }

        with open("logs/predictions.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {"prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))