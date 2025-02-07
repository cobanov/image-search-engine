from typing import List, Optional

from pydantic import BaseModel


class SearchResult(BaseModel):
    filepath: str
    score: float
    scientific_name: Optional[str] = None
    common_name: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
