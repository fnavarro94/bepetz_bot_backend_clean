# In db_queries.py

import logging
from typing import List
from sqlalchemy import text
from common.db_conn import AsyncDBSession

logger = logging.getLogger(__name__)

async def get_all_pet_details_by_user_id(user_id: str) -> List[dict] | None:
    """
    Queries the database for all of a user's pets and returns their basic details.

    Args:
        user_id: The ID of the user to query for.

    Returns:
        A list of dictionaries, each containing a pet's basic details, or None.
    """
    logger.info(f"DB Query: Fetching all pet details for user_id: {user_id}")
    
    # This query now selects only basic info and has no LIMIT.
    query = text("""
        SELECT
            p.name AS pet_name,
            pc.species,
            pc.breed,
            pc.life_stage
        FROM pet p
        JOIN pet_candidate pc ON p.pet_candidate_id = pc.id
        WHERE p.user_id = :user_id;
    """)

    try:
        async with AsyncDBSession() as session:
            async with session.begin():
                result = await session.execute(query, {"user_id": user_id})
                # .all() fetches all matching rows.
                pet_data_rows = result.mappings().all()

        if pet_data_rows:
            logger.info(f"DB Query: Found {len(pet_data_rows)} pets for user '{user_id}'.")
            # Convert the list of RowMappings to a list of standard dicts
            logger.info("La info de las mascotas es %s", pet_data_rows)
            return [dict(row) for row in pet_data_rows]
        else:
            logger.info(f"DB Query: No pet data found for user '{user_id}'.")
            return None

    except Exception as e:
        logger.error(f"DB Query: An error occurred for user_id '{user_id}': {e}")
        return None
