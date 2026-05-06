"""
AutoGuard AI — FastAPI Backend
==============================
REST API for ResNet-50 vehicle damage classification.
Strictly mirrors training notebook (vehicledamage.ipynb) parameters.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Force UTF-8 on Windows ────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (Notebook parity)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["F_Breakage", "F_Crushed", "F_Normal", "R_Breakage", "R_Crushed", "R_Normal"]

MODEL_PATH = Path(__file__).parent / "model_resnetup.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture  (mirrors Notebook Cell 42 — CarClassifierResNetupdated)
# ─────────────────────────────────────────────────────────────────────────────
class CarClassifierResNetupdated(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.model = models.resnet50(weights=None)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.layer4.parameters():
            p.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(0.6153251633150144),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing  (Notebook Cell 3 — ImageNet normalization)
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Model loader  (singleton — loaded once on startup)
# ─────────────────────────────────────────────────────────────────────────────
_model: CarClassifierResNetupdated | None = None


def get_model() -> CarClassifierResNetupdated:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. "
                "Please place 'model_resnetup.pth' next to main.py."
            )
        net = CarClassifierResNetupdated(num_classes=6)
        state = torch.load(str(MODEL_PATH), map_location=torch.device("cpu"), weights_only=False)
        net.load_state_dict(state)
        net.eval()
        _model = net
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# Business logic helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_assessment(label: str) -> dict:
    l = label.lower()
    if "crushed" in l:
        return {
            "severity":    "Critical",
            "cost":        "₹15,000 – ₹40,000",
            "accent":      "#ef4444",
            "glow":        "rgba(239,68,68,0.35)",
            "badge_bg":    "rgba(239,68,68,0.15)",
            "badge_text":  "#fca5a5",
            "icon":        "🚨",
        }
    elif "breakage" in l:
        return {
            "severity":    "Moderate",
            "cost":        "₹5,000 – ₹15,000",
            "accent":      "#f59e0b",
            "glow":        "rgba(245,158,11,0.35)",
            "badge_bg":    "rgba(245,158,11,0.15)",
            "badge_text":  "#fcd34d",
            "icon":        "⚠️",
        }
    else:
        return {
            "severity":    "Minor",
            "cost":        "₹0 – ₹5,000",
            "accent":      "#22c55e",
            "glow":        "rgba(34,197,94,0.35)",
            "badge_bg":    "rgba(34,197,94,0.15)",
            "badge_text":  "#86efac",
            "icon":        "✅",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Response schema
# ─────────────────────────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    label:      str
    confidence: float          # percentage 0–100
    all_probs:  dict[str, float]
    severity:   str
    cost:       str
    accent:     str
    glow:       str
    badge_bg:   str
    badge_text: str
    icon:       str


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AutoGuard AI — Vehicle Damage API",
    description="ResNet-50 6-class vehicle damage classifier. Notebook-synchronized parameters.",
    version="2.0.0",
)

# CORS — allow all origins so the Streamlit frontend can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def preload_model():
    """Warm up model on server start so first request is fast."""
    try:
        get_model()
        print("✅ Model loaded successfully on startup.")
    except FileNotFoundError as e:
        print(f"⚠️  WARNING: {e}")


@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "AutoGuard AI", "version": "2.0.0"}


@app.post("/predict", response_model=PredictResponse, summary="Predict vehicle damage")
async def predict(file: UploadFile = File(...)):
    """
    Upload a vehicle image (JPEG / PNG / WEBP / BMP) and receive:
    - Damage class label
    - Confidence percentage
    - Per-class probabilities
    - Severity level and estimated repair cost
    """
    # Validate content type
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   "Please upload JPEG, PNG, WEBP, or BMP.",
        )

    # Read & decode image
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {exc}")

    # Fix EXIF orientation (smartphone photos)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Preprocess
    tensor = TRANSFORM(image).unsqueeze(0)

    # Inference
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits[0], dim=0)
        top_p, top_i = torch.max(probs, 0)

    label      = CLASS_NAMES[top_i.item()]
    confidence = round(top_p.item() * 100, 2)
    all_probs  = {
        CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }
    assessment = get_assessment(label)

    return PredictResponse(
        label=label,
        confidence=confidence,
        all_probs=all_probs,
        **assessment,
    )
