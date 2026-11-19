"""Download the photos from Kaggle. Run: python -m src.acquire_data"""

import shutil
from pathlib import Path

import kagglehub

from src.preprocessing import CLASS_NAMES, TEST_DIR, TRAIN_DIR

DATASET = "prajwalbhamere/car-damage-severity-dataset"

# Kaggle names the folders training and validation.
# We use validation as our test photos.
SPLITS = {"training": TRAIN_DIR, "validation": TEST_DIR}


def find_folder(root, name):
    """Find a folder by name inside another folder."""
    matches = [p for p in Path(root).rglob(name) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"No '{name}' folder found in {root}")
    return matches[0]


def copy_split(source_dir, target_dir):
    """Copy the photos. Folder 01-minor is saved as minor."""
    for class_name in CLASS_NAMES:
        source = find_folder(source_dir, f"*{class_name}")
        target = target_dir / class_name
        shutil.copytree(source, target, dirs_exist_ok=True)
        print(f"{target}: {len(list(target.glob('*')))} images")


def main():
    download_dir = kagglehub.dataset_download(DATASET)
    for split_name, target_dir in SPLITS.items():
        copy_split(find_folder(download_dir, split_name), target_dir)


if __name__ == "__main__":
    main()