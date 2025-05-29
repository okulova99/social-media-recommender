import pandas as pd
import numpy as np
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class FeatureProcessor:
    def __init__(self):
        self.features = [
            'topic', 'views', 'view_reach', 'TotalTfIdf', 'MaxTfIdf', 'MeanTfIdf',
            'TextCluster', 'DistanceTo1thCluster', 'DistanceTo2thCluster', 'DistanceTo3thCluster',
            'DistanceTo4thCluster', 'DistanceTo5thCluster', 'DistanceTo6thCluster', 'DistanceTo7thCluster',
            'DistanceTo8thCluster', 'DistanceTo9thCluster', 'DistanceTo10thCluster', 'DistanceTo11thCluster',
            'DistanceTo12thCluster', 'DistanceTo13thCluster', 'DistanceTo14thCluster', 'DistanceTo15thCluster',
            'country', 'gender', 'age', 'city', 'exp_group', 'request_hour', 'request_day_of_week',
            'request_month', 'request_week', 'is_weekend', 'time_of_day'
        ]
        self.cat_features = [
            'topic', 'TextCluster', 'gender', 'country', 'city', 'exp_group',
            'time_of_day', 'is_weekend', 'request_week', 'request_month',
            'request_day_of_week', 'request_hour'
        ]
    
    def prepare_features(self, user_data: pd.Series, posts_data: pd.DataFrame, request_time: datetime) -> pd.DataFrame:
        """Подготавливает фичи для предсказания"""
        try:
            # Преобразование времени в Timestamp
            if not isinstance(request_time, pd.Timestamp):
                request_time = pd.Timestamp(request_time)
            
            logger.info(f"Type of request_time: {type(request_time)}")
            logger.info(f"Request time value: {request_time}")
            
            # Безопасное получение номера недели
            iso_calendar = request_time.isocalendar()
            week_number = iso_calendar[1]
            
            # Создаем копию данных постов
            df = posts_data.copy()
            
            # Добавляем временные фичи
            df['request_hour'] = request_time.hour
            df['request_day_of_week'] = request_time.dayofweek
            df['request_month'] = request_time.month
            df['request_week'] = week_number
            df['is_weekend'] = (df['request_day_of_week'] >= 5).astype(int)
            
            # Добавляем фичи пользователя - ИСПРАВЛЕННАЯ ЧАСТЬ
            required_user_features = ['country', 'gender', 'age', 'city', 'exp_group']
            user_dict = user_data.to_dict()
            
            for feature in required_user_features:
                if feature in user_dict:
                    df[feature] = user_dict[feature]
                else:
                    logger.warning(f"User feature {feature} not found in user data")
            
            # Добавляем time_of_day
            bins = [0, 6, 12, 18, 24]
            labels = ['night', 'morning', 'afternoon', 'evening']
            df['time_of_day'] = pd.cut(
                df['request_hour'], bins=bins, labels=labels, 
                include_lowest=True, right=False
            ).fillna('night')
            
            # Проверяем наличие всех фич
            missing = [f for f in self.features if f not in df.columns]
            if missing:
                logger.warning(f"Missing features: {missing}")
            
            # Гарантируем наличие всех необходимых фич
            for feature in self.features:
                if feature not in df.columns:
                    logger.warning(f"Creating dummy column for missing feature: {feature}")
                    df[feature] = 0  # или подходящее значение по умолчанию
                    
            return df[self.features]
        
        except Exception as e:
            logger.exception(f"Critical error in feature preparation: {str(e)}")
            raise