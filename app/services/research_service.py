from datetime import datetime

from fastapi import HTTPException

from agent.crew import LnmResearcher


def sanitize_name(name: str) -> str:
    """Sanitize a name by removing special characters."""
    return "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()


async def get_research(research_table, scientific_name: str, common_name: str) -> dict:
    """Get research results for a species, either from cache or by performing new research."""

    # Sanitize input names
    safe_scientific = sanitize_name(scientific_name)
    safe_common = sanitize_name(common_name)

    print(f"Sanitized names - Scientific: {safe_scientific}, Common: {safe_common}")

    if not safe_scientific or not safe_common:
        raise HTTPException(
            status_code=400,
            detail="Both scientific and common names are required and must contain valid characters",
        )

    # Check cache
    cached_results = (
        research_table.search()
        .where(
            f"scientific_name = '{safe_scientific}' AND common_name = '{safe_common}'"
        )
        .to_pandas()
    )

    if not cached_results.empty:
        print("Found cached research result")
        return {"result": cached_results.iloc[0]["research_result"]}

    # Perform new research
    print("No cache found, performing new research")
    result_text = await perform_research(safe_scientific, safe_common)

    # Cache the result
    research_table.add(
        [
            {
                "scientific_name": safe_scientific,
                "common_name": safe_common,
                "research_result": result_text,
                "timestamp": datetime.now().isoformat(),
            }
        ]
    )

    return {"result": result_text}


async def perform_research(scientific_name: str, common_name: str) -> str:
    """Perform research for a species using LnmResearcher."""
    topic = f"{scientific_name} ({common_name})"
    print(f"Generated topic: {topic}")

    try:
        result = (
            LnmResearcher()
            .crew()
            .kickoff(
                inputs={
                    "topic": topic,
                    "scientific_name": scientific_name,
                    "common_name": common_name,
                }
            )
        )
    except Exception as crew_error:
        print(f"Crew error: {str(crew_error)}")
        raise HTTPException(
            status_code=500, detail=f"Error in research process: {str(crew_error)}"
        )

    # Extract the result text
    if hasattr(result, "raw"):
        return result.raw
    elif hasattr(result, "output"):
        return result.output
    return str(result)
