import os
import logging
from dotenv import load_dotenv
from src.utils.data_loader import DataLoader
from src.utils.model_loader import load_model
from src.utils.feature_processor import FeatureProcessor
from src.utils.recommendation_service import RecommendationService
from fastapi import Depends

logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env файла
load_dotenv()

def get_db_url() -> str:
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    
    if not all([db_user, db_password, db_host, db_port, db_name]):
        logger.error("One or more database environment variables are missing!")
        raise ValueError("Database configuration is incomplete")
    
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

def get_data_loader() -> DataLoader:
    """Зависимость для загрузчика данных"""
    db_url = get_db_url()
    loader = DataLoader(db_url)
    loader.load_features()
    logger.info("Data features loaded")
    return loader

def get_model():
    """Зависимость для ML модели"""
    model_path = os.getenv("MODEL_PATH", "catboost_min_features.cbm")
    model = load_model(model_path)
    logger.info("ML model loaded")
    return model

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