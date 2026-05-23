"""
Single FastAPI app: one upload → damage parts + vehicle identity.

Run from capstone-ai/:
  py -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

POST /detect  (multipart field: file)
"""
from __future__ import annotations

import io
import os
import sys
import pickle
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

# --- Paths ------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DAMAGE_WEIGHTS = ROOT / "CarDamageDetector" / "Weights" / "best.pt"
CAR_MODEL_DIR = ROOT / "CarModelDetector"
PRICE_MODEL_PATH = ROOT / "SparePartPredictor" / "price_model.pkl"
PRICE_ENCODER_DIR = ROOT / "SparePartPredictor" / "price_encoders"

# --- YOLO Config ------------------------------------------------------
DAMAGE_CLASS_LABELS = [
    "Front-Windscreen-Damage",
    "Headlight-Damage",
    "Rear-windscreen-Damage",
    "Sidemirror-Damage",
    "Taillight-Damage",
    "bonnet-dent",
    "boot-dent",
    "doorouter-dent",
    "fender-dent",
    "front-bumper-dent",
    "quaterpanel-dent",
    "rear-bumper-dent",
]
CONF_HIGH = 0.4
CONF_LOW = 0.2

# --- Global Model Holders ------------------------------------------------------
_damage_model: YOLO | None = None
_vehicle_mod = None
_price_model = None
_price_encoders: dict = {}

# --- Vehicle Module Loader ------------------------------------------------------
def _load_vehicle_module():
    path_str = str(CAR_MODEL_DIR.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    prev = os.getcwd()
    try:
        os.chdir(CAR_MODEL_DIR)
        import car_model_detector_v2 as mod

        return mod
    finally:
        os.chdir(prev)

# --- Lifespan: Load all models once at startup ------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _damage_model, _vehicle_mod, _price_model, _price_encoders

    # YOLO weights
    if not DAMAGE_WEIGHTS.is_file():
        raise RuntimeError(f"Missing YOLO weights: {DAMAGE_WEIGHTS}")
    
    # ResNet50 weights
    vehicle_weights = CAR_MODEL_DIR / "best_model.pth"
    if not vehicle_weights.is_file():
        raise RuntimeError(f"Missing vehicle weights: {vehicle_weights}")
    
    # RF model + encoders
    if not PRICE_MODEL_PATH.is_file():
        raise RuntimeError(f"Missing price model: {PRICE_MODEL_PATH}")
    
    _damage_model = YOLO(str(DAMAGE_WEIGHTS))
    _vehicle_mod = _load_vehicle_module()

    with open(PRICE_MODEL_PATH, "rb") as f:
        _price_model = pickle.load(f)
    
    for col in ["make", "model", "year_range", "part_name", "part_condition"]:
        encoder_path = PRICE_ENCODER_DIR / f"{col}.json"
        if not encoder_path.is_file():
            raise RuntimeError(f"Missing encoder: {encoder_path}")
        with open(encoder_path, "r") as f:
            _price_encoders[col] = json.load(f)
    yield

# --- App -------------------------------------------------------
app = FastAPI(title="Capstone AI", version="1.0.0", lifespan=lifespan)

# --- Schemas -------------------------------------------------------
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

# --- YOLO Helpers -------------------------------------------------------
def run_damage_detection(img_bgr: np.ndarray) -> dict:
    results = _damage_model(img_bgr)
    detections = []

    for r in results:
        for box in r.boxes:
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

            if conf >= CONF_HIGH:
                detections.append(
                    {
                        "class": DAMAGE_CLASS_LABELS[cls],
                        "confidence": conf,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                )
            elif conf >= CONF_LOW:
                detections.append(
                    {
                        "class": "unknown-damage",
                        "confidence": conf,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                )

    status = "no_damage" if not detections else "damage_detected"
    return {"status": status, "detections": detections}

# --- RF Helpers -------------------------------------------------------
def _encode(col: str, value: str) -> int:
    mapping = _price_encoders[col]
    if value not in mapping:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown Value '{value}' for '{col}'. Valid: {list(mapping.keys())}"
        )
    return mapping[value]

def _predict_single(make, model_name, year_range, part_name, condition) -> float:
    features = [[
        _encode("make",           make),
        _encode("model",          model_name),
        _encode("year_range",     year_range),
        _encode("part_name",      part_name),
        _encode("part_condition", condition),
    ]]
    return round(float(_price_model.predict(features)[0]), 2)

def _predict_part(make, model_name, year_range, part_name) -> dict:
    return {
        "part_name":     part_name,
        "original_new":  _predict_single(make, model_name, year_range, part_name, "original_new"),
        "original_used": _predict_single(make, model_name, year_range, part_name, "original_used"),
        "aftermarket":   _predict_single(make, model_name, year_range, part_name, "aftermarket"),
    }

# --- API Endpoints -------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "damage_weights": DAMAGE_WEIGHTS.is_file(),
        "vehicle_weights": (CAR_MODEL_DIR / "best_model.pth").is_file(),
        "price_model": PRICE_MODEL_PATH.is_file(),
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Input:  multipart image file
    Output: damage detections + vehicle make/model/year
    """
    contents = await file.read()

    img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    damage = run_damage_detection(img_bgr)

    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    if _vehicle_mod is None:
        raise HTTPException(status_code=503, detail="Vehicle model not loaded.")

    vehicle = _vehicle_mod.predict_pil(pil)

    return {"damage": damage, "vehicle": vehicle}

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
    if _price_model is None:
        raise HTTPException(status_code=503, detail="Price model not loaded.")

    results = []
    for part in request.damaged_parts:
        results.append(_predict_part(
            request.make,
            request.model_name,
            request.year_range,
            part
        ))
    return results