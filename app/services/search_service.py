import io
import os

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.models import SearchResponse, SearchResult
from app.config import DATASET_PATH


async def search_images(
    file: UploadFile, limit: int, table, extractor
) -> SearchResponse:
    """Search for similar images using the provided image file."""
    try:
        # Read and validate the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Extract features
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
