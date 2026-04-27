# Car Damage Detector

A computer vision capstone project that uses a fine-tuned YOLOv8 model to detect and classify car damage from images. The model identifies 12 damage categories and draws annotated bounding boxes with confidence scores.

## Detected Damage Classes

| Category | Category |
|---|---|
| Front Windscreen Damage | Rear Windscreen Damage |
| Headlight Damage | Taillight Damage |
| Sidemirror Damage | Bonnet Dent |
| Boot Dent | Door Outer Dent |
| Fender Dent | Front Bumper Dent |
| Quarter Panel Dent | Rear Bumper Dent |

## Setup

**Requirements:** Python 3.11.9

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The trained model file (`best.pt`, ~88 MB) is excluded from git. Place it at `CarDamageDetector/Weights/best.pt` before running.

## Usage

```bash
python CarDamageDetector/car-damage-detector.py
```

This runs inference on the hardcoded test image (`Media/dent_2.jpg`) and opens a window showing the annotated result. Press any key to close the window.

To test a different image, change `image_path` in `car-damage-detector.py`:

```python
image_path = r"CarDamageDetector\Media\dent_5.jpg"
```

## Project Structure

```
capstone-ai/
├── CarDamageDetector/
│   ├── car-damage-detector.py   # Main inference script
│   ├── Weights/
│   │   └── best.pt              # Trained YOLOv8 model (not tracked in git)
│   └── Media/
│       └── dent_1.jpg ... dent_80.jpg   # Test images
├── requirements.txt
└── README.md
```

## Model Details

- **Architecture:** YOLOv8 (via Ultralytics)
- **Training environment:** Google Colab
- **Confidence threshold:** 0.4 (detections below this are discarded)
- **Input:** Any image containing a car
- **Output:** Bounding boxes labeled with damage class and confidence score
