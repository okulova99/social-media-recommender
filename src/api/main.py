from fastapi import FastAPI, HTTPException, Query, Depends
from . import schemas
from .dependencies import get_recommendation_service
from datetime import datetime
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Social Media Recommendation System")

@app.get("/post/recommendations/", response_model=List[schemas.PostGet])
def recommended_posts(
    id: int = Query(..., example=201),
    time: datetime = Query(..., example="2021-10-15T12:00:00Z"),
    limit: int = Query(5, example=5),
    recommendation_service = Depends(get_recommendation_service)
) -> List[schemas.PostGet]:
    """Возвращает персонализированные рекомендации постов"""
    try:
        return recommendation_service(
            user_id=id,
            request_time=time,
            limit=limit
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "ok", "message": "Service is operational"}