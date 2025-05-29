import os
from catboost import CatBoostClassifier
import logging
from typing import Union

logger = logging.getLogger(__name__)



def get_model_path(path: str) -> str:
    """Определяет путь к модели в зависимости от окружения"""
    if os.environ.get("IS_LMS") == "1":
        return '/workdir/user_input/model'
    return path

def load_model(model_path: Union[str, None] = None) -> CatBoostClassifier:
    """Загружает CatBoost модель"""
    try:
        # Если путь не указан, используем значение из переменных окружения
        if model_path is None:
            model_path = os.getenv("MODEL_PATH", "catboost_min_features.cbm")
        
        final_path = get_model_path(model_path)
        logger.debug(f"Loading model from: {final_path}")
        
        # Загружаем как CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(final_path)
        
        logger.info(f"Model loaded successfully from {final_path}")
        return model
    
    except FileNotFoundError as e:
        logger.critical(f"Model file not found: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Error loading model: {str(e)}")  # Логируем traceback
        raise