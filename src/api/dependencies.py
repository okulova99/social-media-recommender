import os
import logging
from dotenv import load_dotenv
from src.utils.data_loader import DataLoader
from src.utils.model_loader import load_model
from src.utils.feature_processor import FeatureProcessor
from src.utils.recommendation_service import RecommendationService
from fastapi import Depends

logger = logging.getLogger(__name__)

load_dotenv()

def get_db_url() -> str:
    """Формирует URL для подключения к БД из переменных окружения"""
    return f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}" \
           f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

def get_data_loader() -> DataLoader:
    """Зависимость для загрузчика данных"""
    loader = DataLoader(get_db_url())
    loader.load_features()
    return loader

def get_model():
    """Зависимость для ML модели"""
    return load_model(os.getenv("MODEL_PATH", "catboost_model.cbm"))

def get_feature_processor() -> FeatureProcessor:
    """Зависимость для обработки признаков"""
    return FeatureProcessor()

def get_recommendation_service(
    data_loader: DataLoader = Depends(get_data_loader),
    model = Depends(get_model),
    feature_processor: FeatureProcessor = Depends(get_feature_processor)
) -> RecommendationService:
    """Зависимость для сервиса рекомендаций"""
    return RecommendationService(
        data_loader=data_loader,
        model=model,
        feature_processor=feature_processor
    )