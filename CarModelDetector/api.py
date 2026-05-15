from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

from car_model_detector_v2 import predict_pil, build_reference, SUPPORTED_MAKES, _centroids

app = FastAPI()


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    return predict_pil(img)


@app.post("/build-reference")
async def build_reference_endpoint(image_dir: str = "test_media"):
    """
    Build feature centroids from images in image_dir. Run once after startup
    to enable OOD detection. Uses the test_media folder by default.
    """
    build_reference(image_dir)
    return {
        "status": "ok",
        "makes_with_centroids": list(_centroids.keys()),
        "supported_makes": SUPPORTED_MAKES,
    }


@app.get("/supported-makes")
async def supported_makes():
    return {
        "supported_makes": SUPPORTED_MAKES,
        "ood_detection_active": len(_centroids) > 0,
        "makes_with_centroids": list(_centroids.keys()),
    }
