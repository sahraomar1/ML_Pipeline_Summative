"""Simple numbers about the photos, for each damage level.

brightness: how light the photo is
contrast: how different the light and dark parts are
edges: how many sharp lines there are (broken metal has many)
"""

import numpy as np

from src.preprocessing import CLASS_NAMES, TRAIN_DIR, list_image_files, load_images


def image_features(image):
    """Measure one photo."""
    gray = image.mean(axis=2)
    edges_down, edges_across = np.gradient(gray)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "edges": float(np.hypot(edges_down, edges_across).mean()),
    }


def class_features(folder, max_images=150):
    """Average the numbers for one damage level."""
    files = list_image_files(folder)
    images, _ = load_images(files[:max_images])
    features = [image_features(image) for image in images]
    summary = {"count": len(files)}
    for name in ["brightness", "contrast", "edges"]:
        values = [f[name] for f in features]
        summary[name] = round(float(np.mean(values)), 2) if values else 0.0
    return summary


def dataset_insights(data_dir=TRAIN_DIR):
    """Do this for every damage level."""
    return {name: class_features(data_dir / name) for name in CLASS_NAMES}