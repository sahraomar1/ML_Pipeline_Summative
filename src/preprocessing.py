"""Get the photos ready for the model."""

import io
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
CLASS_NAMES = ["minor", "moderate", "severe"]
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"

MODEL_PATH = ROOT_DIR / "models" / "damage_model.h5"


def load_image(source):
    """Open one photo and make it 160 x 160."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    image = Image.open(source).convert("RGB").resize(IMAGE_SIZE)
    return np.asarray(image, dtype="float32")


def load_images(paths):
    """Open many photos. Skip the ones that are broken."""
    images, good_paths = [], []
    for path in paths:
        try:
            images.append(load_image(path))
            good_paths.append(path)
        except OSError:
            print(f"Skipped unreadable image: {path}")
    return np.array(images), good_paths


def list_image_files(folder):
    """Get all the photo files in a folder."""
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in IMAGE_TYPES)


def augment(train_data):
    """Flip and turn the training photos a little."""
    change = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.05),
    ])
    return train_data.map(lambda images, labels: (change(images, training=True), labels))


def load_train_and_validation(train_dir=TRAIN_DIR, validation_split=0.2, seed=42):
    """Split the train folder: 80% to train, 20% to check."""
    options = dict(
        directory=train_dir,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=validation_split,
        seed=seed,
    )
    train_data = keras.utils.image_dataset_from_directory(subset="training", **options)
    validation_data = keras.utils.image_dataset_from_directory(subset="validation", **options)
    return augment(train_data), validation_data


def load_test_data(test_dir=TEST_DIR):
    """Load the test photos."""
    return keras.utils.image_dataset_from_directory(
        test_dir,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )