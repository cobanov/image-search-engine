import utils

# Load configuration
config = utils.load_config("./configs/ocean_clip.yml")

# VectorDB Configs
LANCE_DB_PATH = config["LANCEDB"]
COLLECTION_NAME = config["COLLECTION_NAME"]
RESEARCH_COLLECTION = config["RESEARCH_COLLECTION"]

# Model Configs
MODEL_NAME = config["MODEL_NAME"]
MODEL_DIM = config["MODEL_DIM"]

# Dataset Configs
DATASET_PATH = config["DATASET_PATH"]
