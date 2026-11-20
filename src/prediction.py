"""Predict the damage level of one photo."""

import numpy as np

from src.model import MODEL_PATH, load_saved_model
from src.preprocessing import CLASS_NAMES, load_image

CLAIM_ROUTES = {
    "minor": "Fast track - approve a repair quote automatically.",
    "moderate": "Send to a claims adjuster for review.",
    "severe": "Escalate to a senior adjuster - possible total loss.",
}

_cache = {"model": None, "file_time": None}


def get_model():
    """Load the model once. Load it again only if the file changed."""
    file_time = MODEL_PATH.stat().st_mtime
    if _cache["model"] is None or _cache["file_time"] != file_time:
        _cache["model"] = load_saved_model()
        _cache["file_time"] = file_time
    return _cache["model"]


def predict_image(source):
    """Predict one photo and say where the claim should go."""
    image = load_image(source)
    batch = np.expand_dims(image, axis=0)
    probabilities = get_model().predict(batch, verbose=0)[0]

    label = CLASS_NAMES[int(probabilities.argmax())]
    return {
        "label": label,
        "confidence": float(probabilities.max()),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probabilities)},
        "claim_route": CLAIM_ROUTES[label],
    }