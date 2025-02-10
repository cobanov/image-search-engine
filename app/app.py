import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import engine.db_utils as db_utils
from app.config import (
    LANCE_DB_PATH,
    COLLECTION_NAME,
    RESEARCH_COLLECTION,
    MODEL_NAME,
    MODEL_DIM,
    DATASET_PATH,
)
from app.models import SearchResponse
from app.services.research_service import get_research
from app.services.search_service import search_images
from engine.encoder import FeatureExtractor
from engine.batch_encoder import CLIPBatchFeatureExtractor

load_dotenv()

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
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Mount the dataset directory
app.mount("/images", StaticFiles(directory=DATASET_PATH), name="images")

# Initialize CLIP extractor for both image and text search
clip_extractor = CLIPBatchFeatureExtractor()

# Connect to LanceDB
db = db_utils.get_lancedb_client(LANCE_DB_PATH)
table = db_utils.open_table(db, COLLECTION_NAME)
research_table = db_utils.open_table(db, RESEARCH_COLLECTION)


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")


@app.post("/research")
async def research(scientific_name: str = Form(...), common_name: str = Form(...)):
    return await get_research(research_table, scientific_name, common_name)


@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...), limit: int = 25
):  # Changed limit to 25 for 5x5 grid
    return await search_images(file, limit, table, clip_extractor)


@app.post("/search/text", response_model=SearchResponse)
async def search_by_text(query: str = Form(...), limit: int = 25):
    """Search images using text query with CLIP"""
    # Encode the text query
    text_features = clip_extractor.encode_text([query])

    # Search the database using the text features
    results = table.search(text_features[0].numpy()).limit(limit).to_list()

    # Format results
    search_results = []
    for result in results:
        search_results.append(
            {
                "filepath": f"/images/{result['id']}",
                "score": float(result.get("_distance", result.get("score", 0.0))),
                "scientific_name": result.get("scientific_name", ""),
                "common_name": result.get("common_name", ""),
            }
        )

    return SearchResponse(results=search_results, total_results=len(search_results))


if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("app/static", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
