import io
import os

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.models import SearchResponse, SearchResult
from app.config import DATASET_PATH


async def search_images(
    file: UploadFile, limit: int, table, extractor
) -> SearchResponse:
    """Search for similar images using the provided image file and CLIP."""
    try:
        # Read and validate the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Extract features using CLIP
        image_features = extractor.extract_features([image])

        # Search in the database using the CLIP features
        results = table.search(image_features[0].numpy()).limit(limit).to_list()

        # Format results
        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    filepath=f"/images/{result['id']}",
                    score=float(result.get("_distance", result.get("score", 0.0))),
                    scientific_name=result.get("scientific_name", ""),
                    common_name=result.get("common_name", ""),
                )
            )

        return SearchResponse(results=search_results, total_results=len(search_results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
