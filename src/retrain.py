"""Retrain the model on the photos users uploaded."""

import numpy as np

from src import database
from src.model import retrain_model
from src.preprocessing import CLASS_NAMES, load_images

AUTO_RETRAIN_THRESHOLD = 50


def prepare_uploads(uploads):
    """Open the new photos and turn the labels into numbers."""
    labels_by_path = {row["file_path"]: row["label"] for row in uploads}
    images, good_paths = load_images(list(labels_by_path))
    labels = np.array([CLASS_NAMES.index(labels_by_path[path]) for path in good_paths])
    return images, labels


def should_auto_retrain():
    """Yes if enough new photos are waiting and no training is running."""
    enough_data = len(database.get_new_uploads()) >= AUTO_RETRAIN_THRESHOLD
    return enough_data and not is_running()


def is_running():
    last_run = database.get_last_run()
    return bool(last_run) and last_run["status"] == "running"


def run_retraining():
    """Retrain once and save the result in the database."""
    run_id = database.start_run()
    uploads = database.get_new_uploads()
    try:
        images, labels = prepare_uploads(uploads)
        if len(images) == 0:
            raise ValueError("No new images to train on. Upload some first.")
        retrain_model(images, labels)
        database.mark_uploads_used([row["id"] for row in uploads])
        database.finish_run(run_id, "done", len(images), "Model retrained and saved.")
    except Exception as error:
        database.finish_run(run_id, "failed", 0, str(error))