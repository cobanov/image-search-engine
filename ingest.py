import os
from glob import glob

import torch
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

import engine.db_utils as db_utils
import utils
from engine.batch_encoder import BatchFeatureExtractor, ImageDataset

config = utils.load_config("./configs/ocean_resnet50_v2.yml")

COLLECTION_NAME = config["COLLECTION_NAME"]
LANCE_DB_PATH = config["LANCEDB"]
MODEL_NAME = config["MODEL_NAME"]
MODEL_DIM = config["MODEL_DIM"]
DATASET_PATH = config["DATASET_PATH"]


print(f"\n🔍 Scanning for images in: {DATASET_PATH}")
image_paths = []
for ext in ["*.JPEG", "*.jpeg", "*.jpg", "*.png", "*.webp"]:
    found = glob(os.path.join(DATASET_PATH, "**/" + ext), recursive=True)
    image_paths.extend(found)
    if found:
        print(f"Found {len(found)} {ext} images")

db = db_utils.get_lancedb_client(LANCE_DB_PATH)
table = db_utils.create_table(db, COLLECTION_NAME, dim=MODEL_DIM)
# table = db_utils.open_table(db, COLLECTION_NAME)

extractor = BatchFeatureExtractor(MODEL_NAME, batch_size=256, num_workers=0)


def extract_features_generator(extractor, image_paths):
    dataset = ImageDataset(image_paths, extractor.preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=extractor.batch_size,
        shuffle=False,
        num_workers=extractor.num_workers,
        pin_memory=False,  # Optimized for GPU
    )

    pbar = tqdm(
        total=len(image_paths),
        desc="📸 Processing Images",
        unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    with torch.no_grad():
        for batch_images, batch_paths in dataloader:
            batch_images = batch_images.to(extractor.device, non_blocking=True)
            output = extractor.model(batch_images)  # Forward pass
            feature_vectors = output.cpu().numpy()  # Move results to CPU
            feature_vectors = normalize(
                feature_vectors, norm="l2"
            )  # Normalize features
            pbar.update(len(batch_paths))
            yield [
                {"vector": vector.tolist(), "filepath": path}
                for vector, path in zip(feature_vectors, batch_paths)
            ]


if __name__ == "__main__":
    table.add(extract_features_generator(extractor, image_paths))
    print(f"\n✅ Processing complete!")
