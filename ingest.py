import os
from glob import glob

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pyarrow as pa

import engine.db_utils as db_utils
import utils
from engine.batch_encoder import CLIPBatchFeatureExtractor, ImageDataset

config = utils.load_config("./configs/ocean_clip.yml")

COLLECTION_NAME = config["COLLECTION_NAME"]
RESEARCH_COLLECTION = config["RESEARCH_COLLECTION"]  # Add research collection name
LANCE_DB_PATH = config["LANCEDB"]
DATASET_PATH = config["DATASET_PATH"]
MODEL_NAME = config["MODEL_NAME"]
MODEL_DIM = config["MODEL_DIM"]

# Set test mode for quick testing
TEST_MODE = False
TEST_LIMIT = 500

print(f"\n🔍 Scanning for images in: {DATASET_PATH}")
image_paths = []
for ext in ["*.JPEG", "*.jpeg", "*.jpg", "*.png", "*.webp"]:
    found = glob(os.path.join(DATASET_PATH, "**/" + ext), recursive=True)
    image_paths.extend(found)
    if found:
        print(f"Found {len(found)} {ext} images")

# Limit dataset size for testing
if TEST_MODE:
    print(f"\n🧪 Test mode: Limiting dataset to {TEST_LIMIT} images")
    image_paths = image_paths[:TEST_LIMIT]

db = db_utils.get_lancedb_client(LANCE_DB_PATH)

# Create main table for image search
print("\n📦 Creating main search table...")
table = db_utils.create_table(db, COLLECTION_NAME, dim=MODEL_DIM)

# Create research cache table
print("\n📦 Creating research cache table...")
research_schema = pa.schema(
    [
        pa.field("scientific_name", pa.string()),
        pa.field("common_name", pa.string()),
        pa.field("result", pa.string()),
    ]
)
research_table = db.create_table(
    RESEARCH_COLLECTION, schema=research_schema, mode="overwrite"
)

# Initialize CLIP extractor
extractor = CLIPBatchFeatureExtractor(
    model_name=MODEL_NAME,
    batch_size=32,  # Smaller batch size for testing
    num_workers=0,
)


def extract_features_generator(extractor, image_paths):
    dataset = ImageDataset(image_paths, extractor.preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=extractor.batch_size,
        shuffle=False,
        num_workers=extractor.num_workers,
        pin_memory=True,
    )

    pbar = tqdm(
        total=len(image_paths),
        desc="📸 Processing Images",
        unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    with torch.no_grad():
        for batch_images, batch_paths in dataloader:
            # Extract features using CLIP
            batch_features = extractor.extract_features(batch_images)
            # Features are already normalized by CLIP
            feature_vectors = batch_features.numpy()

            pbar.update(len(batch_paths))
            yield [
                {
                    "vector": vector.tolist(),
                    "filepath": path,
                    "id": os.path.basename(path),
                    "scientific_name": "",  # Empty string for now
                    "common_name": "",  # Empty string for now
                }
                for vector, path in zip(feature_vectors, batch_paths)
            ]


if __name__ == "__main__":
    print("\n🚀 Starting feature extraction and database ingestion...")
    table.add(extract_features_generator(extractor, image_paths))
    print(f"\n✅ Processing complete! Added {len(image_paths)} images to the database.")
    print(f"✅ Created research cache table: {RESEARCH_COLLECTION}")
