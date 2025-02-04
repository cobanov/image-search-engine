import os
from glob import glob

import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import db_utils
import utils
from batch_encoder import BatchFeatureExtractor, ImageDataset
from encoder import FeatureExtractor

config = utils.load_config("./configs/animals.yml")

COLLECTION_NAME = config["COLLECTION_NAME"]
LANCE_DB_PATH = config["LANCEDB"]
MODEL_NAME = config["MODEL_NAME"]
MODEL_DIM = config["MODEL_DIM"]
DATASET_PATH = config["DATASET_PATH"]


image_encoder = FeatureExtractor(MODEL_NAME)


db = db_utils.get_lancedb_client(LANCE_DB_PATH)
table = db_utils.open_table(db, COLLECTION_NAME)


image_paths = glob(os.path.join(DATASET_PATH, "**/*.JPEG"))
print(f"Found {len(image_paths)} images")


# # Sinle Image Embedding
# for i, filepath in enumerate(tqdm(image_paths, desc="Generating embeddings ...")):
#     try:
#         image_embedding = image_encoder(filepath)
#         data = [{"vector": image_embedding, "filepath": filepath}]
#         table.add(data)

#     except Exception as e:
#         print(
#             f"Skipping file: {filepath} due to an error occurs during the embedding process:\n{e}"
#         )
#         continue

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

    with torch.no_grad():
        for batch_images, batch_paths in dataloader:
            batch_images = batch_images.to(extractor.device, non_blocking=True)
            output = extractor.model(batch_images)  # Forward pass
            feature_vectors = output.cpu().numpy()  # Move results to CPU
            feature_vectors = normalize(
                feature_vectors, norm="l2"
            )  # Normalize features

            yield [
                {"vector": vector.tolist(), "filepath": path}
                for vector, path in zip(feature_vectors, batch_paths)
            ]


if __name__ == "__main__":
    for i, batch_data in enumerate(
        tqdm(
            extract_features_generator(extractor, image_paths),
            desc="Generating embeddings ...",
        )
    ):
        table.add(batch_data)
        if i == 0:
            break
    print("Done")
