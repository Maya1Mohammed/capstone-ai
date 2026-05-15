import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import sys
import os

# Config
MODEL_PATH = "best_model.pth"
ENCODER_DIR = "encoders"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load encoders
def load_encoder(name):
    with open(os.path.join(ENCODER_DIR, f"{name}.json"), "r") as f:
        return json.load(f)

enc_make = load_encoder("make")
enc_model = load_encoder("model")
enc_year_range = load_encoder("year_range")

NUM_MAKES = len(enc_make)
NUM_MODELS = len(enc_model)
NUM_YEAR_RANGES = len(enc_year_range)

# Model Definition (must match training architecture)
class MultiHeadResNet50(nn.Module):
    def __init__(self, num_makes, num_models, num_year_ranges):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0,4),
        )
        self.head_make = nn.Linear(512, num_makes)
        self.head_model = nn.Linear(512, num_models)
        self.head_year_range = nn.Linear(512, num_year_ranges)

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared_fc(x)
        return self.head_make(x), self.head_model(x), self.head_year_range(x)

# Load Model
model = MultiHeadResNet50(NUM_MAKES, NUM_MODELS, NUM_YEAR_RANGES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out_make, out_model, out_year = model(tensor)
    
    pred_make = enc_make[str(out_make.argmax(1).item())]
    pred_model = enc_model[str(out_model.argmax(1).item())]
    pred_year = enc_year_range[str(out_year.argmax(1).item())]

    conf_make = torch.softmax(out_make, dim=1).max().item() * 100
    conf_model = torch.softmax(out_model, dim=1).max().item() * 100
    conf_year = torch.softmax(out_year, dim=1).max().item() * 100

    print(f"\nImage: {image_path}")
    print(f"  Make:       {pred_make:<15} ({conf_make:.1f}% confidence)")
    print(f"  Model:      {pred_model:<15} ({conf_model:.1f}% confidence)")
    print(f"  Year Range: {pred_year:<15} ({conf_year:.1f}% confidence)")

# Run on Image Passed as Argument, or Test Folder
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single image: python car_model_detector.py car_image.jpg
        predict(sys.argv[1])
    else:
        # Test on all images in "test_media" folder
        test_dir = "test_media"
        if os.path.exists(test_dir):
            for fname in os.listdir(test_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    predict(os.path.join(test_dir, fname))
        else:
            print("Usage: python test_model.py <image_path>")
            print("Or create a 'test_images/' folder with images to batch test.")