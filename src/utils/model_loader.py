import os
from catboost import CatBoostClassifier
import logging
from typing import Union

logger = logging.getLogger(__name__)



def get_model_path(path: str) -> str:
    """Определяет путь к модели в зависимости от окружения (LMS или локальное)"""
    if os.environ.get("IS_LMS") == "1":
        return '/workdir/user_input/model'
    return path

def load_model(model_path: Union[str, None] = None) -> CatBoostClassifier:
    """Загружает CatBoost модель из указанного пути"""
    try:
        model_path = model_path or os.getenv("MODEL_PATH", "catboost_min_features.cbm")
        final_path = get_model_path(model_path)
        
        model = CatBoostClassifier()
        model.load_model(final_path)
        
        logger.info(f"Model loaded from: {final_path}")
        return model
    
    except FileNotFoundError:
        logger.critical("Model file not found")
        raise
    except Exception:
        logger.exception("Model loading failed")
        raise