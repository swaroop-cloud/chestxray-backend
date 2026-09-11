"""
Preprocesses an uploaded X-ray and runs it through the model,
applying the per-disease thresholds selected during training.
"""

import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image

from model_loader import get_model

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

# Fallback if no threshold file is found — override with your real
# validation-optimized thresholds via outputs/disease_thresholds.csv
DEFAULT_THRESHOLDS = {label: 0.5 for label in DISEASE_LABELS}

IMG_SIZE = (224, 224)


def load_thresholds(csv_path: Path | None = None) -> dict:
    """
    Reads a csv with columns: disease,threshold
    (this matches the disease_thresholds.csv your notebook already exports).
    Falls back to 0.5 for any disease not found in the file.
    """
    thresholds = dict(DEFAULT_THRESHOLDS)
    if not csv_path or not Path(csv_path).exists():
        return thresholds

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = row.get("disease") or row.get("Disease")
            value = row.get("threshold") or row.get("Threshold")
            if disease in thresholds and value is not None:
                thresholds[disease] = float(value)
    return thresholds


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Grayscale/RGB PNG or JPEG -> (1, 224, 224, 3) float32 array in [0, 1]."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(image_bytes: bytes, thresholds: dict | None = None) -> dict:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    model = get_model()
    batch = preprocess_image(image_bytes)
    raw_probs = model.predict(batch, verbose=0)[0]

    results = []
    for label, prob in zip(DISEASE_LABELS, raw_probs):
        threshold = thresholds.get(label, 0.5)
        results.append({
            "disease": label,
            "probability": round(float(prob), 4),
            "threshold": round(float(threshold), 4),
            "positive": bool(prob >= threshold),
        })

    results.sort(key=lambda r: r["probability"], reverse=True)
    return {"predictions": results}
