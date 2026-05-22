# Capstone AI — Car Analysis System

A computer vision capstone project that analyzes car images to detect damage, identify the vehicle make/model/year, and estimate spare part prices. Three specialized ML models are integrated into a single unified FastAPI service.

---

## Architecture Overview

```
Image Upload
     │
     ▼
┌─────────────────────────────────────┐
│         Unified API (main.py)       │
│  POST /detect  ·  GET /health       │
└──────────┬──────────────────┬───────┘
           │                  │
           ▼                  ▼
  ┌─────────────────┐  ┌──────────────────┐
  │ Car Damage      │  │ Car Model        │
  │ Detector        │  │ Detector         │
  │ (YOLOv8)        │  │ (ResNet50)       │
  └────────┬────────┘  └──────────────────┘
           │
           ▼
  ┌─────────────────┐
  │ Spare Part      │
  │ Predictor       │
  │ (Random Forest) │
  └─────────────────┘
```

---

## Models

### 1. Car Damage Detector

Uses a fine-tuned **YOLOv8** model to detect and classify car damage. Returns bounding boxes and confidence scores for each detected damage region.

#### Detected Damage Classes (12)

| Category | Category |
|---|---|
| Front Windscreen Damage | Rear Windscreen Damage |
| Headlight Damage | Taillight Damage |
| Sidemirror Damage | Bonnet Dent |
| Boot Dent | Door Outer Dent |
| Fender Dent | Front Bumper Dent |
| Quarter Panel Dent | Rear Bumper Dent |

#### Model Details

- **Architecture:** YOLOv8 (via Ultralytics)
- **Weights file:** `CarDamageDetector/Weights/best.pt` (~88 MB, not tracked in git)
- **Confidence thresholds:** ≥ 0.4 → named class, 0.2–0.4 → `unknown-damage`, < 0.2 → discarded
- **Output:**
  ```json
  {
    "status": "damage_detected",
    "detections": [
      {
        "class": "front-bumper-dent",
        "confidence": 0.85,
        "bbox": { "x1": 100, "y1": 50, "x2": 200, "y2": 150 }
      }
    ]
  }
  ```

#### Standalone usage

```bash
# Script
python CarDamageDetector/car-damage-detector-v2.py

# API (from CarDamageDetector/)
uvicorn api:app --reload
# POST /detect  →  multipart image upload
```

---

### 2. Car Model Detector

Uses a fine-tuned **multi-head ResNet50** to predict car make, model, and year range. Includes out-of-distribution (OOD) detection so unsupported vehicles return `"Unknown"` instead of a forced wrong prediction.

#### Supported Vehicles

| Make | Model | Year Ranges |
|------|-------|-------------|
| changan | estar | 2010–2012, 2013–2016, 2017, 2018–2020, 2020–2026 |
| ford | fusion | 2010–2012, 2013–2016, 2017, 2018–2020, 2020–2026 |
| volkswagen | id4 | 2010–2012, 2013–2016, 2017, 2018–2020, 2020–2026 |

#### OOD Detection

A frozen **ImageNet-pretrained ResNet50** acts as a general visual encoder. For each prediction, the cosine similarity between the input's features and the stored centroid per make is computed. If the maximum similarity across all makes falls below **0.75**, the result is flagged as `"Unknown"`.

Build reference centroids once before running (requires images in `CarModelDetector/test_media/`):

```bash
# From CarModelDetector/
python car_model_detector_v2.py --build-reference
```

#### Model Details

- **Architecture:** Multi-head ResNet50 (shared backbone → three output heads)
- **Heads:** make (3 classes), model (3 classes), year range (5 classes)
- **Temperature scaling:** T = 2.0 (calibrates overconfident softmax scores)
- **Confidence thresholds:** make ≥ 45%, model ≥ 40%, year range ≥ 45%
- **Weights file:** `CarModelDetector/best_model.pth` (~98 MB, not tracked in git)
- **OOD centroids:** `CarModelDetector/reference_centroids.npz` (not tracked in git, build locally)
- **Output:**
  ```json
  {
    "make": "ford",
    "model": "fusion",
    "year_range": "2013_2016",
    "confidence": { "make": 78.5, "model": 92.1, "year_range": 45.0 },
    "supported_makes": ["changan", "ford", "volkswagen"]
  }
  ```

#### Standalone usage

```bash
# Single image
python CarModelDetector/car_model_detector_v2.py path/to/image.jpg

# Batch (all images in test_media/)
python CarModelDetector/car_model_detector_v2.py

# API (from CarModelDetector/)
uvicorn api:app --reload
# POST /detect           →  multipart image upload
# POST /build-reference  →  rebuild OOD centroids from test_media/
# GET  /supported-makes  →  list supported makes and OOD status
```

---

### 3. Spare Part Predictor

Uses a **Random Forest** model to estimate the price of damaged spare parts in three condition tiers: original new, original used, and aftermarket.

#### Input Features

| Feature | Options |
|---------|---------|
| Make | changan, ford, volkswagen |
| Model | estar, fusion, id4 |
| Year range | 2010–2012, 2013–2016, 2017, 2018–2020, 2020–2026 |
| Part name | 12 damage categories (same as damage detector) |
| Part condition | original_new, original_used, aftermarket |

#### Model Details

- **Architecture:** Random Forest (scikit-learn)
- **Model file:** `SparePartPredictor/price_model.pkl` (~3 MB)
- **Encoders:** `SparePartPredictor/price_encoders/` (JSON label encoders for each feature)
- **Output:**
  ```json
  [
    {
      "part_name": "front-bumper-dent",
      "original_new": 250.0,
      "original_used": 70.0,
      "aftermarket": 45.0
    }
  ]
  ```

#### Standalone usage

```bash
# API (from SparePartPredictor/)
uvicorn api:app --reload
# POST /estimate  →  JSON body with make, model, year_range, detections
# GET  /health    →  health check
```

---

## Unified API (main.py)

The primary entry point. Loads all three models on startup and exposes a single endpoint that runs damage detection and vehicle identification in one call. The spare part predictor is called automatically using the vehicle and damage results.

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns status of all model weight files |
| `POST` | `/detect` | Accepts a multipart image; returns damage + vehicle + price estimates |

### Response shape

```json
{
  "damage": {
    "status": "damage_detected",
    "detections": [...]
  },
  "vehicle": {
    "make": "ford",
    "model": "fusion",
    "year_range": "2013_2016",
    "confidence": { "make": 78.5, "model": 92.1, "year_range": 45.0 },
    "supported_makes": ["changan", "ford", "volkswagen"]
  },
  "spare_parts": [
    {
      "part_name": "front-bumper-dent",
      "original_new": 250.0,
      "original_used": 70.0,
      "aftermarket": 45.0
    }
  ]
}
```

---

## Setup

**Requirements:** Python 3.11.9

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### Model weights (not tracked in git)

| File | Size | Where to place |
|------|------|----------------|
| `best.pt` | ~88 MB | `CarDamageDetector/Weights/best.pt` |
| `best_model.pth` | ~98 MB | `CarModelDetector/best_model.pth` |
| `reference_centroids.npz` | small | build with `--build-reference` (see above) |
| `price_model.pkl` | ~3 MB | `SparePartPredictor/price_model.pkl` |

---

## Project Structure

```
capstone-ai/
├── main.py                             # Unified FastAPI app (all three models)
├── requirements.txt
├── README.md
│
├── CarDamageDetector/
│   ├── car-damage-detector.py          # v1 standalone script
│   ├── car-damage-detector-v2.py       # v2 standalone script (confidence thresholds)
│   ├── api.py                          # Standalone FastAPI service
│   ├── Weights/
│   │   └── best.pt                     # YOLOv8 weights (not tracked in git)
│   └── Media/
│       └── dent_1.jpg … dent_80.jpg    # Demo / training images
│
├── CarModelDetector/
│   ├── car_model_detector.py           # v1 standalone script
│   ├── car_model_detector_v2.py        # v2 standalone script (OOD detection)
│   ├── api.py                          # Standalone FastAPI service
│   ├── best_model.pth                  # ResNet50 weights (not tracked in git)
│   ├── reference_centroids.npz         # OOD centroids (not tracked in git)
│   ├── encoders/
│   │   ├── make.json
│   │   ├── model.json
│   │   └── year_range.json
│   └── test_media/                     # Images used to build OOD centroids
│
└── SparePartPredictor/
    ├── spare_part_predictor.py         # Core prediction logic
    ├── api.py                          # Standalone FastAPI service
    ├── price_model.pkl                 # Random Forest model
    └── price_encoders/
        ├── make.json
        ├── model.json
        ├── part_name.json
        ├── part_condition.json
        └── year_range.json
```

---

## Technology Stack

| Category | Libraries |
|----------|-----------|
| Deep learning | PyTorch, torchvision, Ultralytics (YOLOv8) |
| Traditional ML | scikit-learn (Random Forest) |
| Image processing | OpenCV, Pillow, scikit-image |
| Web framework | FastAPI, Uvicorn |
| Data | NumPy, Pandas, SciPy |
| Visualization | Matplotlib, Seaborn |
