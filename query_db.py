import db_utils
import utils
from encoder import FeatureExtractor

config = utils.load_config("./configs/animals_siglip.yml")

COLLECTION_NAME = config["COLLECTION_NAME"]
LANCE_DB_PATH = config["LANCEDB"]
MODEL_NAME = config["MODEL_NAME"]
MODEL_DIM = config["MODEL_DIM"]
DATASET_PATH = config["DATASET_PATH"]

# connect to LanceDB
db = db_utils.get_lancedb_client(LANCE_DB_PATH)
table = db_utils.open_table(db, COLLECTION_NAME)
table.create_index("l2", vector_column_name="vector")


extractor = FeatureExtractor(MODEL_NAME)

image = "./dataset/reverse_image_search/train/bannister/n02788148_10587.JPEG"
query = extractor(image)
result = table.search(query)

df = result.to_pandas()
df.drop(columns=["vector"], inplace=True)
df.to_csv("results/result.csv", index=False)
