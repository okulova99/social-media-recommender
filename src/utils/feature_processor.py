import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FeatureProcessor:
    def __init__(self):
        # Основные фичи
        self.features = [
            'topic', 'views', 'view_reach', 'TotalTfIdf', 'MaxTfIdf', 'MeanTfIdf',
            'TextCluster', 'DistanceTo1thCluster', 'DistanceTo2thCluster', 'DistanceTo3thCluster',
            'DistanceTo4thCluster', 'DistanceTo5thCluster', 'DistanceTo6thCluster', 'DistanceTo7thCluster',
            'DistanceTo8thCluster', 'DistanceTo9thCluster', 'DistanceTo10thCluster', 'DistanceTo11thCluster',
            'DistanceTo12thCluster', 'DistanceTo13thCluster', 'DistanceTo14thCluster', 'DistanceTo15thCluster',
            'country', 'gender', 'age', 'city', 'exp_group', 'request_hour', 'request_day_of_week',
            'request_month', 'request_week', 'is_weekend', 'time_of_day'
        ]

        # Категориальные фичи
        self.cat_features = [
            'topic', 'TextCluster', 'gender', 'country', 'city', 'exp_group',
            'time_of_day', 'is_weekend', 'request_week', 'request_month',
            'request_day_of_week', 'request_hour'
        ]
    
    def prepare_features(self, user_data: pd.Series,
                         posts_data: pd.DataFrame,
                         request_time: datetime) -> pd.DataFrame:
        """Создает финальный датафрейм признаков для предсказания"""
        try:
            # Копируем данные для безопасности
            df = posts_data.copy()
            request_time = pd.Timestamp(request_time)
            
            # Добавляем временнЫе фичи
            time_features = {
                'request_hour': request_time.hour,
                'request_day_of_week': request_time.dayofweek,
                'request_month': request_time.month,
                'request_week': request_time.isocalendar()[1],
                'is_weekend': int(request_time.dayofweek >= 5)
            }
            df = df.assign(**time_features)
            
            # Добавляем пользовательские фичи
            user_features = ['country', 'gender', 'age', 'city', 'exp_group']            
            for feature in user_features:
                df[feature] = user_data.get(feature, None)
            
            # Добавляем время суток
            time_bins = [0, 6, 12, 18, 24]
            time_labels = ['night', 'morning', 'afternoon', 'evening']
            df['time_of_day'] = pd.cut(df['request_hour'], bins=time_bins,
                                       labels=time_labels, right=False).fillna('night')
            
            # Гарантируем наличие всех фичей
            for feature in self.features:
                if feature not in df.columns:
                    df[feature] = 0  # Значение по умолчанию
                    
            return df[self.features]
        
        except Exception:
            logger.exception("Feature preparation error")
            raise