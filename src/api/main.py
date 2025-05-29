from fastapi import FastAPI, HTTPException, Query, Depends
from . import schemas
from .dependencies import get_recommendation_service
from datetime import datetime
import logging
from typing import List

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/post/recommendations/", response_model=List[schemas.PostGet])
def recommended_posts(
    id: int = Query(..., example=123),
    time: datetime = Query(..., example="2023-10-15T12:00:00Z"),
    limit: int = Query(5, example=5),
    recommendation_service = Depends(get_recommendation_service)
) -> List[schemas.PostGet]:
    try:
        return recommendation_service.get_recommendations(
            user_id=id,
            request_time=time,
            limit=limit
        )
        
    except ValueError as e:
        logger.warning(f"User not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Service is running"}