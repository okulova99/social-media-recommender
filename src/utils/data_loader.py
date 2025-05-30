import pandas as pd
from sqlalchemy import create_engine
import os
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.user_features = None
        self.post_features = None
        self.liked_posts = None
        self.post_details = {}  # Кэш для быстрого доступа
    
    def batch_load_sql(self, query: str) -> pd.DataFrame:
        """Загружает данные из PostgreSQL с пакетной обработкой"""
        CHUNKSIZE = 200000
        chunks = []
        with self.engine.connect().execution_options(stream_results=True) as conn:
            for chunk in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
                chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)
    
    def load_features(self):
        """Загружает все необходимые данные для рекомендаций"""
        try:            
            # Загрузка фичей постов
            self.post_features = self.batch_load_sql(
                "SELECT * FROM d_okulova_post_features_lesson_22"
            )
            if self.post_features.empty:
                raise ValueError("Post features are empty")
            
            # Загрузка фичей пользователей
            self.user_features = self.batch_load_sql(
                "SELECT * FROM d_okulova_user_features_lesson_22"
            )
            if self.user_features.empty:
                raise ValueError("User features are empty")
                
            # Загрузка лайков
            self.liked_posts = self.batch_load_sql(
                "SELECT DISTINCT post_id, user_id FROM public.feed_data WHERE action='like'"
            )
            
            # Создаем кэш постов
            self.post_details = self.post_features.set_index('post_id').to_dict('index')
            
            logger.info("Data loading completed successfully")
            return True
            
        except Exception:
            logger.exception("Data loading failed")
            raise