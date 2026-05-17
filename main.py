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
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DAMAGE_WEIGHTS = ROOT / "CarDamageDetector" / "Weights" / "best.pt"
CAR_MODEL_DIR = ROOT / "CarModelDetector"

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

_damage_model: YOLO | None = None
_vehicle_mod = None


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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _damage_model, _vehicle_mod

    if not DAMAGE_WEIGHTS.is_file():
        raise RuntimeError(f"Missing YOLO weights: {DAMAGE_WEIGHTS}")

    vehicle_weights = CAR_MODEL_DIR / "best_model.pth"
    if not vehicle_weights.is_file():
        raise RuntimeError(f"Missing vehicle weights: {vehicle_weights}")

    _damage_model = YOLO(str(DAMAGE_WEIGHTS))
    _vehicle_mod = _load_vehicle_module()

    yield


app = FastAPI(title="Capstone AI", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "damage_weights": DAMAGE_WEIGHTS.is_file(),
        "vehicle_weights": (CAR_MODEL_DIR / "best_model.pth").is_file(),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Model 1: damaged exterior parts.
    Model 2: make, model, year_range.
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
        raise HTTPException(status_code=503, detail="Vehicle model not loaded")

    vehicle = _vehicle_mod.predict_pil(pil)

    return {
        "damage": damage,
        "vehicle": vehicle,
    }
