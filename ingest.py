import os
from glob import glob

from tqdm import tqdm

import db_utils
import utils
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


for i, filepath in enumerate(tqdm(image_paths, desc="Generating embeddings ...")):
    try:
        image_embedding = image_encoder(filepath)
        data = [{"vector": image_embedding, "filepath": filepath}]
        table.add(data)

    except Exception as e:
        print(
            f"Skipping file: {filepath} due to an error occurs during the embedding process:\n{e}"
        )
        continue
