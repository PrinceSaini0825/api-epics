"""
api/main.py – FastAPI server for Fire Detection inference.

Endpoints:
    GET  /              -> health check
    GET  /models        -> list available models
    POST /predict       -> predict from uploaded image
    POST /predict/base64 -> predict from base64 image string

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from models import HybridFireDetector, ModelA, ModelB

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "./weights")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD   = 0.37
CLASS_NAMES = ["fire_images", "non_fire_images"]

# ── Transform (same as test_transform in training) ────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Model registry ────────────────────────────────────────────────────────────
MODELS: dict[str, torch.nn.Module] = {}


def load_models():
    """Load all available .pth weights from WEIGHTS_DIR into memory."""
    model_configs = {
        "model_a": (ModelA,  "best_ModelA.pth"),
        "model_b": (ModelB,  "best_ModelB.pth"),
        "hybrid":  (None,    "best_Hybrid.pth"),   # special handling
    }

    for key, (cls, filename) in model_configs.items():
        path = os.path.join(WEIGHTS_DIR, filename)
        if not os.path.exists(path):
            print(f"⚠️  Weight file not found, skipping: {path}")
            continue
        try:
            if key == "hybrid":
                model = HybridFireDetector()
                model.load_state_dict(torch.load(path, map_location=DEVICE))
            else:
                model = cls()
                model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            MODELS[key] = model
            print(f"✅ Loaded {key} from {path}")
        except Exception as e:
            print(f"❌ Failed to load {key}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    MODELS.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fire Detection API",
    description="Classify images as fire or non-fire using PyTorch CNN models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    model_used:  str
    label:       str
    is_fire:     bool
    confidence:  float
    threshold:   float


class Base64Request(BaseModel):
    image_base64: str
    model: Literal["model_a", "model_b", "hybrid"] = "hybrid"


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_inference(model: torch.nn.Module, img: Image.Image, model_name: str) -> PredictionResponse:
    tensor = infer_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = model(tensor).item()
    # ImageFolder sorts classes alphabetically:
    #   class 0 = fire_images, class 1 = non_fire_images
    # The model outputs P(class=1) = P(non_fire).
    # So: prob > threshold  →  non_fire   |   prob <= threshold  →  fire
    is_fire = prob <= THRESHOLD
    label   = CLASS_NAMES[0] if is_fire else CLASS_NAMES[1]   # fire_images or non_fire_images
    fire_confidence = 1.0 - prob if is_fire else prob    # always show confidence in predicted class
    return PredictionResponse(
        model_used=model_name,
        label=label,
        is_fire=is_fire,
        confidence=round(fire_confidence, 4),
        threshold=THRESHOLD,
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "loaded_models": list(MODELS.keys()),
    }


@app.get("/models", tags=["Info"])
def list_models():
    return {"available_models": list(MODELS.keys())}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Image file (jpg, png, webp)"),
    model: Literal["model_a", "model_b", "hybrid"] = "hybrid",
):
    """Upload an image file and get a fire/non-fire prediction."""
    if model not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not loaded. Available: {list(MODELS.keys())}")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid jpg/png/webp file.")

    return run_inference(MODELS[model], img, model)


@app.post("/predict/base64", response_model=PredictionResponse, tags=["Inference"])
def predict_base64(body: Base64Request):
    """Send a base64-encoded image string and get a fire/non-fire prediction."""
    if body.model not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{body.model}' not loaded.")

    try:
        raw = base64.b64decode(body.image_base64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    return run_inference(MODELS[body.model], img, body.model)
