import pandas as pd
from sqlalchemy import create_engine
import os
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(self.db_url)
        self.user_features = None
        self.post_features = None
        self.liked_posts = None
        self.post_details_dict = None  # Кэш для деталей постов
    
    def batch_load_sql(self, query: str) -> pd.DataFrame:
        """Пакетная загрузка данных из SQL"""
        CHUNKSIZE = 200000
        conn = self.engine.connect().execution_options(stream_results=True)
        chunks = []
        try:
            for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
                chunks.append(chunk_dataframe)
            return pd.concat(chunks, ignore_index=True)
        finally:
            conn.close()
    
    def load_features(self):
        """Загружает все необходимые фичи"""
        try:
            logger.info("Starting feature loading...")
            
            # Загружаем посты
            logger.info("Loading post features...")
            self.post_features = self.batch_load_sql("SELECT * FROM d_okulova_post_features_lesson_22")
            
            # Проверка типа и содержимого
            if not isinstance(self.post_features, pd.DataFrame):
                logger.error(f"post_features is not DataFrame! Type: {type(self.post_features)}")
                raise TypeError("post_features must be DataFrame")
                
            if self.post_features.empty:
                logger.error("Post features are empty!")
                raise ValueError("Post features are empty")
            else:
                logger.info(f"Loaded post features: {self.post_features.shape[0]} rows, {self.post_features.shape[1]} columns")
            
            # Загружаем пользователей
            logger.info("Loading user features...")
            self.user_features = self.batch_load_sql("SELECT * FROM d_okulova_user_features_lesson_22")
            
            if not isinstance(self.user_features, pd.DataFrame):
                logger.error(f"user_features is not DataFrame! Type: {type(self.user_features)}")
                raise TypeError("user_features must be DataFrame")
                
            if self.user_features.empty:
                logger.error("User features are empty!")
                raise ValueError("User features are empty")
            else:
                logger.info(f"Loaded user features: {self.user_features.shape[0]} rows, {self.user_features.shape[1]} columns")
            
            # Загружаем лайки
            logger.info("Loading liked posts...")
            self.liked_posts = self.batch_load_sql(
                "SELECT DISTINCT post_id, user_id FROM public.feed_data WHERE action='like'"
            )
            
            if not isinstance(self.liked_posts, pd.DataFrame):
                logger.error(f"liked_posts is not DataFrame! Type: {type(self.liked_posts)}")
                raise TypeError("liked_posts must be DataFrame")
                
            if self.liked_posts.empty:
                logger.warning("Liked posts are empty!")
            else:
                logger.info(f"Loaded liked posts: {self.liked_posts.shape[0]} rows")
            
            # Создаем словарь для быстрого доступа к деталям постов
            logger.info("Creating post details dictionary...")
            self.post_details_dict = self.post_features.set_index('post_id').to_dict('index')
            
            logger.info("Features loaded successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Critical error loading features: {str(e)}")
            # Сбрасываем все данные при ошибке
            self.user_features = None
            self.post_features = None
            self.liked_posts = None
            self.post_details_dict = None
            raise RuntimeError(f"Failed to load features: {str(e)}") from e
    
    def get_user_features(self, user_id: int) -> pd.Series:
        """Возвращает фичи пользователя"""
        return self.user_features[self.user_features['user_id'] == user_id].iloc[0]
    
    def get_post_features(self) -> pd.DataFrame:
        """Возвращает фичи постов"""
        return self.post_features
    
    def get_liked_posts(self) -> pd.DataFrame:
        """Возвращает уже лайкнутые посты"""
        return self.liked_posts