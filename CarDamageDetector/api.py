from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import math
from ultralytics import YOLO

app = FastAPI()
model = YOLO(r"CarDamageDetector\Weights\best.pt")

CLASS_LABELS = [
    "Front-Windscreen-Damage", "Headlight-Damage", "Rear-windscreen-Damage",
    "Sidemirror-Damage", "Taillight-Damage", "bonnet-dent",
    "boot-dent", "doorouter-dent", "fender-dent",
    "front-bumper-dent", "quaterpanel-dent", "rear-bumper-dent"
]

CONF_HIGH = 0.4
CONF_LOW  = 0.2


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    results = model(img)
    detections = []

    for r in results:
        for box in r.boxes:
            conf = round(float(box.conf[0]), 2)
            cls  = int(box.cls[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

            if conf >= CONF_HIGH:
                detections.append({
                    "class": CLASS_LABELS[cls],
                    "confidence": conf,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })
            elif conf >= CONF_LOW:
                detections.append({
                    "class": "unknown-damage",
                    "confidence": conf,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

    status = "no_damage" if not detections else "damage_detected"
    return {"status": status, "detections": detections}
