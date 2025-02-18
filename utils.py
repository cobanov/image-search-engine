import os
from glob import glob

import yaml


def load_config(yaml_path="config.yaml"):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)  # Load YAML file into a dictionary
    return config


def scan_for_images(DATASET_PATH):
    print(f"\n🔍 Scanning for images in: {DATASET_PATH}")
    image_paths = []
    for ext in ["*.JPEG", "*.jpeg", "*.jpg", "*.png", "*.webp"]:
        found = glob(os.path.join(DATASET_PATH, "**/" + ext), recursive=True)
        # Normalize paths to use forward slashes
        found = [path.replace("\\", "/") for path in found]
        image_paths.extend(found)
        if found:
            print(f"Found {len(found)} {ext} images")
    return image_paths
