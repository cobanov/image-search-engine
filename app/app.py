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

# Initialize feature extractor
extractor = FeatureExtractor(MODEL_NAME)

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
    return await search_images(file, limit, table, extractor)


if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("app/static", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
