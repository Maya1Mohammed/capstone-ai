import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import json
import sys
import os
import numpy as np

MODEL_PATH = "best_model.pth"
ENCODER_DIR = "encoders"
REFERENCE_PATH = "reference_centroids.npz"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPERATURE = 4.0
CONF_THRESHOLD_MAKE = 60.0
CONF_THRESHOLD_MODEL = 50.0
CONF_THRESHOLD_YEAR = 55.0

# Cosine similarity threshold on ImageNet features.
# OOD cars (unknown make) typically score below 0.80 vs known-make centroids.
COSINE_SIM_THRESHOLD = 0.90


def load_encoder(name):
    with open(os.path.join(ENCODER_DIR, f"{name}.json"), "r") as f:
        return json.load(f)

enc_make = load_encoder("make")
enc_model = load_encoder("model")
enc_year_range = load_encoder("year_range")

NUM_MAKES = len(enc_make)
NUM_MODELS = len(enc_model)
NUM_YEAR_RANGES = len(enc_year_range)

SUPPORTED_MAKES = list(enc_make.values())


# --- Fine-tuned classifier (make/model/year prediction) ---

class MultiHeadResNet50(nn.Module):
    def __init__(self, num_makes, num_models, num_year_ranges):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0, 4),
        )
        self.head_make = nn.Linear(512, num_makes)
        self.head_model = nn.Linear(512, num_models)
        self.head_year_range = nn.Linear(512, num_year_ranges)

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared_fc(x)
        return self.head_make(x), self.head_model(x), self.head_year_range(x)


_classifier = MultiHeadResNet50(NUM_MAKES, NUM_MODELS, NUM_YEAR_RANGES)
_classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
_classifier.to(DEVICE)
_classifier.eval()


# --- Frozen ImageNet backbone for OOD detection ---
# The fine-tuned model collapses all cars into 3 clusters, making its features
# useless for detecting unknown makes. A general ImageNet backbone preserves
# visual differences (body shape, proportions, design) that distinguish car types.

_ood_backbone = nn.Sequential(
    *list(models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-1],
    nn.Flatten(),
)
_ood_backbone.to(DEVICE)
_ood_backbone.eval()


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def _load_centroids():
    if not os.path.exists(REFERENCE_PATH):
        return {}
    data = np.load(REFERENCE_PATH)
    return {
        key: torch.tensor(data[key], dtype=torch.float32, device=DEVICE)
        for key in data.files
    }


_centroids = _load_centroids()


def _classify(tensor: torch.Tensor):
    """Run the fine-tuned classifier. Returns (out_make, out_model, out_year)."""
    with torch.no_grad():
        return _classifier(tensor)


def _ood_features(tensor: torch.Tensor) -> torch.Tensor:
    """Extract 2048-dim features from the frozen ImageNet backbone."""
    with torch.no_grad():
        return _ood_backbone(tensor)  # [1, 2048]


def _calibrated_confidence(logits: torch.Tensor) -> tuple[float, int]:
    scaled = logits / TEMPERATURE
    probs = torch.softmax(scaled, dim=1)
    return probs.max().item() * 100, probs.argmax(1).item()


def _is_ood(features: torch.Tensor, debug: bool = False) -> bool:
    """
    Returns True when the image is out-of-distribution (unknown make).

    Uses the MAX similarity across all class centroids rather than just the
    predicted class. A genuine in-distribution car scores 0.90+ to its own
    centroid; an unknown car spreads its similarity evenly and peaks lower.
    """
    max_sim = 0.0
    for make_name, centroid in _centroids.items():
        sim = F.cosine_similarity(features, centroid.unsqueeze(0)).item()
        if debug:
            print(f"  [OOD] similarity to {make_name}: {sim:.4f} (threshold: {COSINE_SIM_THRESHOLD})")
        max_sim = max(max_sim, sim)

    return max_sim < COSINE_SIM_THRESHOLD


def predict_pil(img: Image.Image, debug: bool = False) -> dict:
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    out_make, out_model, out_year = _classify(tensor)

    conf_make, idx_make = _calibrated_confidence(out_make)
    conf_model, idx_model = _calibrated_confidence(out_model)
    conf_year, idx_year = _calibrated_confidence(out_year)

    make_known = conf_make >= CONF_THRESHOLD_MAKE
    pred_make = enc_make[str(idx_make)] if make_known else "Unknown"

    # OOD gate using ImageNet features — catches confident-but-wrong predictions
    if make_known and _centroids:
        features = _ood_features(tensor)
        if _is_ood(features, debug=debug):
            pred_make = "Unknown"
            make_known = False

    pred_model = (
        enc_model[str(idx_model)]
        if make_known and conf_model >= CONF_THRESHOLD_MODEL
        else "Unknown"
    )
    pred_year = (
        enc_year_range[str(idx_year)]
        if conf_year >= CONF_THRESHOLD_YEAR
        else "Unknown"
    )

    return {
        "make": pred_make,
        "model": pred_model,
        "year_range": pred_year,
        "confidence": {
            "make": round(conf_make, 1),
            "model": round(conf_model, 1),
            "year_range": round(conf_year, 1),
        },
        "supported_makes": SUPPORTED_MAKES,
    }


def predict(image_path: str, debug: bool = False) -> dict:
    return predict_pil(Image.open(image_path).convert("RGB"), debug=debug)


def build_reference(image_dir: str = "test_media"):
    """
    Build per-class feature centroids using the frozen ImageNet backbone.
    Only images where the fine-tuned classifier is >=80% confident (raw softmax)
    contribute to each class centroid.

    Run once before using OOD detection:
        python car_model_detector_v2.py --build-reference
    """
    print(f"Building reference centroids from '{image_dir}'...")
    features_by_make: dict[str, list] = {name: [] for name in enc_make.values()}

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(image_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            # Use raw softmax on the classifier to identify which make this image is
            with torch.no_grad():
                out_make, _, _ = _classifier(tensor)
            probs = torch.softmax(out_make, dim=1)
            conf = probs.max().item() * 100
            idx = probs.argmax(1).item()

            if conf >= 80.0:
                # Store ImageNet features (not fine-tuned features) for the centroid
                feat = _ood_features(tensor).squeeze(0).cpu().numpy()
                make_name = enc_make[str(idx)]
                features_by_make[make_name].append(feat)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    centroids = {}
    for make_name, feats in features_by_make.items():
        if feats:
            centroids[make_name] = np.mean(feats, axis=0).astype(np.float32)
            print(f"  {make_name}: {len(feats)} reference images")
        else:
            print(f"  {make_name}: no high-confidence images found")

    if centroids:
        np.savez(REFERENCE_PATH, **centroids)
        print(f"Saved to '{REFERENCE_PATH}'")
        global _centroids
        _centroids = {
            k: torch.tensor(v, dtype=torch.float32, device=DEVICE)
            for k, v in centroids.items()
        }
    else:
        print("No centroids built.")


if __name__ == "__main__":
    print(f"Supported makes: {', '.join(SUPPORTED_MAKES)}")

    if "--build-reference" in sys.argv:
        build_reference()
        sys.exit(0)

    if not _centroids:
        print("Warning: reference_centroids.npz not found.")
        print("Run 'python car_model_detector_v2.py --build-reference' once to enable OOD detection.")

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    debug = "--debug" in sys.argv

    if args:
        result = predict(args[0], debug=debug)
        print(f"\nImage: {args[0]}")
        print(f"  Make:       {result['make']:<15} ({result['confidence']['make']:.1f}% confidence)")
        print(f"  Model:      {result['model']:<15} ({result['confidence']['model']:.1f}% confidence)")
        print(f"  Year Range: {result['year_range']:<15} ({result['confidence']['year_range']:.1f}% confidence)")
    else:
        test_dir = "test_media"
        if os.path.exists(test_dir):
            for fname in sorted(os.listdir(test_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(test_dir, fname)
                    result = predict(path)
                    print(f"\nImage: {fname}")
                    print(f"  Make:       {result['make']:<15} ({result['confidence']['make']:.1f}%)")
                    print(f"  Model:      {result['model']:<15} ({result['confidence']['model']:.1f}%)")
                    print(f"  Year Range: {result['year_range']:<15} ({result['confidence']['year_range']:.1f}%)")
        else:
            print("Usage: python car_model_detector_v2.py <image_path>")
            print("       python car_model_detector_v2.py --build-reference")
