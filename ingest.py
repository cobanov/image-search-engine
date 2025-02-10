import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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

# Scan for images
image_paths = utils.scan_for_images(DATASET_PATH)

# Limit dataset size for testing
if TEST_MODE:
    print(f"\n🧪 Test mode: Limiting dataset to {TEST_LIMIT} images")
    image_paths = image_paths[:TEST_LIMIT]

# Connect to LanceDB
db = db_utils.get_lancedb_client(LANCE_DB_PATH)

print("\n📦 Creating main search table...")
table = db_utils.create_table(db, COLLECTION_NAME, dim=MODEL_DIM)

if RESEARCH_COLLECTION:
    print("\n📦 Creating research cache table...")
    research_table = db_utils.create_research_table(db, RESEARCH_COLLECTION, dim=MODEL_DIM)


# Initialize CLIP extractor
extractor = CLIPBatchFeatureExtractor(
    model_name=MODEL_NAME,
    batch_size=32,
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
            batch_features = extractor.extract_features(batch_images)
            feature_vectors = batch_features.numpy()

            pbar.update(len(batch_paths))
            yield [
                {
                    "vector": vector.tolist(),
                    "filepath": path,
                    "id": os.path.basename(path),
                }
                for vector, path in zip(feature_vectors, batch_paths)
            ]


if __name__ == "__main__":
    print("\n🚀 Starting feature extraction and database ingestion...")
    table.add(extract_features_generator(extractor, image_paths))
    print(f"\n✅ Processing complete! Added {len(image_paths)} images to the database.")
