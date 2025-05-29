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
        self.data_loader = data_loader
        self.model = model
        self.feature_processor = feature_processor
    
    def get_recommendations(self, user_id: int, request_time: datetime, limit: int = 5) -> List[PostGet]:
        """Возвращает рекомендации для пользователя"""
        try:
            # Упрощенная проверка пользователя
            if user_id not in self.data_loader.user_features['user_id'].values:
                return []
            
            # Получаем данные
            user_data = self.data_loader.get_user_features(user_id)
            posts_data = self.data_loader.get_post_features()
            
            # Готовим фичи
            features = self.feature_processor.prepare_features(user_data, posts_data, request_time)
            
            # Упрощенное предсказание
            features['pred_proba'] = self.model.predict_proba(features.values)[:, 1]
            features['post_id'] = posts_data['post_id'].values

            
            # Фильтрация и возврат результата
            liked_posts = self.data_loader.liked_posts
            user_liked = liked_posts[liked_posts['user_id'] == user_id]['post_id'].values
            new_content = features[~features['post_id'].isin(user_liked)]

            logger.info(f"Total posts before filtering: {len(features)}")
            logger.info(f"User liked posts count: {len(user_liked)}")
            logger.info(f"First 5 liked posts: {user_liked[:5]}")
            logger.info(f"First 5 available posts: {features['post_id'].iloc[:5].tolist()}")

            logger.info(f"Liked posts type: {type(user_liked[0]) if len(user_liked) > 0 else 'empty'}")
            logger.info(f"Features posts type: {type(features['post_id'].iloc[0])}")
            
            top_posts = new_content.nlargest(limit, 'pred_proba')['post_id'].tolist()

            logger.info(f"Prediction stats: min={features['pred_proba'].min()}, max={features['pred_proba'].max()}, mean={features['pred_proba'].mean()}")
            
            return [
                PostGet(
                    id=pid,
                    text=self.data_loader.post_details_dict[pid]['text'],
                    topic=self.data_loader.post_details_dict[pid]['topic']
                )
                for pid in top_posts
                if pid in self.data_loader.post_details_dict
            ]
        
        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            return []