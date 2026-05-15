# Capstone AI — Car Analysis Models

A computer vision capstone project with two models: a car damage detector and a car make/model/year classifier, each exposed as a FastAPI service.

---

## Car Damage Detector

Uses a fine-tuned YOLOv8 model to detect and classify car damage from images. Identifies 12 damage categories with annotated bounding boxes and confidence scores.

### Detected Damage Classes

| Category | Category |
|---|---|
| Front Windscreen Damage | Rear Windscreen Damage |
| Headlight Damage | Taillight Damage |
| Sidemirror Damage | Bonnet Dent |
| Boot Dent | Door Outer Dent |
| Fender Dent | Front Bumper Dent |
| Quarter Panel Dent | Rear Bumper Dent |

### Usage

```bash
# Standalone script
python CarDamageDetector/car-damage-detector-v2.py

# API (run from CarDamageDetector/)
uvicorn api:app --reload
# POST /detect  →  multipart image upload
```

> **Note:** The trained model file (`best.pt`, ~88 MB) is excluded from git. Place it at `CarDamageDetector/Weights/best.pt` before running.

### Model Details

- **Architecture:** YOLOv8 (via Ultralytics)
- **Confidence thresholds:** 0.4 high / 0.2 low (below low → discarded, between → `unknown-damage`)
- **Output:** `{ "status": "damage_detected" | "no_damage", "detections": [...] }`

---

## Car Model Detector

Uses a fine-tuned multi-head ResNet50 to predict car **make**, **model**, and **year range** from an image. Includes out-of-distribution (OOD) detection to return `"Unknown"` for car makes not seen during training instead of forcing a wrong prediction.

### Supported Makes

`changan`, `ford`, `volkswagen`

### OOD Detection

The detector uses a frozen ImageNet-pretrained ResNet50 as a general visual encoder. Before returning a prediction, it checks the cosine similarity between the input image's features and the stored centroid for each known make. If the maximum similarity across all centroids falls below **0.90**, the car is flagged as `"Unknown"` rather than misclassified.

Build the reference centroids once before running predictions:

```bash
# From CarModelDetector/
python car_model_detector_v2.py --build-reference
```

### Usage

```bash
# Single image
python CarModelDetector/car_model_detector_v2.py path/to/image.jpg

# Batch (all images in test_media/)
python CarModelDetector/car_model_detector_v2.py

# API (run from CarModelDetector/)
uvicorn api:app --reload
# POST   /detect           →  multipart image upload
# POST   /build-reference  →  rebuilds OOD centroids from test_media/
# GET    /supported-makes  →  lists supported makes and OOD status
```

> **Note:** The trained model file (`best_model.pth`) is excluded from git. Place it at `CarModelDetector/best_model.pth` before running.

### Model Details

- **Architecture:** Multi-head ResNet50 (shared backbone, three output heads)
- **Heads:** make classification, model classification, year-range classification
- **Temperature scaling:** T = 4.0 (calibrates overconfident softmax scores)
- **Output:** `{ "make", "model", "year_range", "confidence": { ... }, "supported_makes": [...] }`

---

## Setup

**Requirements:** Python 3.11.9

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## Project Structure

```
capstone-ai/
├── CarDamageDetector/
│   ├── car-damage-detector-v2.py   # Standalone inference script
│   ├── api.py                      # FastAPI service
│   ├── Weights/
│   │   └── best.pt                 # YOLOv8 weights (not tracked in git)
│   └── Media/
│       └── dent_1.jpg ... dent_80.jpg
├── CarModelDetector/
│   ├── car_model_detector_v2.py    # Standalone inference script (with OOD detection)
│   ├── api.py                      # FastAPI service
│   ├── encoders/
│   │   ├── make.json
│   │   ├── model.json
│   │   └── year_range.json
│   ├── best_model.pth              # ResNet50 weights (not tracked in git)
│   ├── reference_centroids.npz     # OOD centroids (not tracked in git, build locally)
│   └── test_media/
├── requirements.txt
└── README.md
```
