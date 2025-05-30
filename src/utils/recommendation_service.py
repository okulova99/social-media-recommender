import pandas as pd
import logging
from datetime import datetime
from src.utils.data_loader import DataLoader
from src.utils.feature_processor import FeatureProcessor
from typing import List
from src.api.schemas import PostGet

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self, data_loader: DataLoader, model, feature_processor: FeatureProcessor):
        self.data = data_loader
        self.model = model
        self.features = feature_processor
    
    def get_recommendations(self, user_id: int,
                            request_time: datetime,
                            limit: int = 5) -> List[PostGet]:
        """Генерирует персонализированные рекомендации постов"""
        try:
            # Проверка существования пользователя
            if user_id not in self.data.user_features['user_id'].values:
                return []
            
            # Получение данных
            user_data = self.data.user_features[
                self.data.user_features['user_id'] == user_id
            ].iloc[0]
            
            # Подготовка признаков
            features = self.features.prepare_features(
                user_data, self.data.post_features, request_time
            )
            
            # Предсказание
            features['pred_proba'] = self.model.predict_proba(features.values)[:, 1]
            features['post_id'] = self.data.post_features['post_id']
            
            # Фильтрация лайкнутых постов
            user_likes = self.data.liked_posts[
                self.data.liked_posts['user_id'] == user_id
                ]['post_id']
            new_posts = features[~features['post_id'].isin(user_likes)]

            # Выбор топ-N постов
            top_posts = new_posts.nlargest(limit, 'pred_proba')['post_id'].tolist()
            
            # Формирование результата
            return [
                PostGet(
                    id=post_id,
                    text=self.data.post_details[post_id]['text'],
                    topic=self.data.post_details[post_id]['topic']
                )
                for post_id in top_posts
                if post_id in self.data.post_details
            ]
        
        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            return []