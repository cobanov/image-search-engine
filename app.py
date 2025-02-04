import io
import os
from typing import List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import db_utils
import utils
from encoder import FeatureExtractor

# Load configuration
config = utils.load_config("./configs/ocean_resnet50_v2.yml")

COLLECTION_NAME = config["COLLECTION_NAME"]
LANCE_DB_PATH = config["LANCEDB"]
MODEL_NAME = config["MODEL_NAME"]
MODEL_DIM = config["MODEL_DIM"]
DATASET_PATH = config["DATASET_PATH"]

# Initialize FastAPI app
app = FastAPI(
    title="Image Search API",
    description="API for finding similar images using image-based search",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the web interface
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount the dataset directory
app.mount("/images", StaticFiles(directory=DATASET_PATH), name="images")

# Initialize feature extractor
extractor = FeatureExtractor(MODEL_NAME)

# Connect to LanceDB
db = db_utils.get_lancedb_client(LANCE_DB_PATH)
table = db_utils.open_table(db, COLLECTION_NAME)


class SearchResult(BaseModel):
    filepath: str
    score: float
    scientific_name: Optional[str] = None
    common_name: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...), limit: int = 25
):  # Changed limit to 25 for 5x5 grid
    try:
        # Read and validate the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Extract features
        # Save image temporarily
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        query_vector = extractor(temp_path)
        os.remove(temp_path)  # Clean up

        # Search in the database
        results = table.search(query_vector).limit(limit).to_pandas()

        # Format results with corrected image paths
        search_results = [
            SearchResult(
                filepath="/images/"
                + os.path.relpath(row["filepath"], DATASET_PATH).replace("\\", "/"),
                score=float(row["_distance"]),
                scientific_name=row.get("scientific_name", None),
                common_name=row.get("common_name", None),
            )
            for _, row in results.iterrows()
        ]

        return SearchResponse(results=search_results, total_results=len(search_results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
