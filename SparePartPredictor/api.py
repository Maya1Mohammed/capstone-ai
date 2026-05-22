import pickle
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Config
MODEL_PATH = "price_model.pkl"
ENCODER_DIR = "price_encoders"

app = FastAPI()

# Load model and encoders once at startup
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

encoders = {}
for col in ["make", "model", "year_range", "part_name", "part_condition"]:
    with open(os.path.join(ENCODER_DIR, f"{col}.json"), "r") as f:
        encoders[col] = json.load(f)

# Request/Response Schemas
class EstimateRequest(BaseModel):
    make: str
    model_name: str
    year_range: str
    damaged_parts: List[str]

class PartResult(BaseModel):
    part_name: str
    original_new: float
    original_used: float
    aftermarket: float

# Helpers
def encode(col, value):
    mapping = encoders[col]
    if value not in mapping:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown Value '{value} for '{col}'. Valid: {list(mapping.keys())}"
        )
    return mapping[value]

def predict_single(make, model_name, year_range, part_name, condition):
    features = [[
        encode("make", make),
        encode("model", model_name),
        encode("year_range", year_range),
        encode("part_name", part_name),
        encode("part_condition", condition)
    ]]
    return round(float(model.predict(features)[0]), 2)

def predict_part(make, model_name, year_range, part_name):
    return {
        "part_name": part_name,
        "original_new": predict_single(make, model_name, year_range, part_name, "original_new"),
        "original_used": predict_single(make, model_name, year_range, part_name, "original_used"),
        "aftermarket": predict_single(make, model_name, year_range, part_name, "aftermarket")
    }

# Endpoint
@app.post("/estimate", response_model=List[PartResult])
def estimate(request: EstimateRequest):
    """
    Input:
    {
        "make":          "ford",
        "model_name":    "fusion",
        "year_range":    "2013_2016",
        "damaged_parts": ["front-bumper-dent", "Headlight-Damage", "bonnet-dent"]
    }

    Output:
    [
        { "part_name": "front-bumper-dent", "original_new": 250.0, "original_used": 70.0, "aftermarket": 45.0 },
        { "part_name": "Headlight-Damage",  "original_new": 239.8, "original_used": 62.4, "aftermarket": 52.8 },
        { "part_name": "bonnet-dent",       "original_new": 399.5, "original_used": 102.3, "aftermarket": 84.6 }
    ]
    """
    results = []
    for part in request.damaged_parts:
        result = predict_part(
            request.make,
            request.model_name,
            request.year_range,
            part
        )
        results.append(result)
    return results

# Health Check
@app.get("/health")
def health():
    return {"status": "ok"}