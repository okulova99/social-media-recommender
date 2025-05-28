# Рекомендательная система постов

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.85-green)](https://fastapi.tiangolo.com)

## Описание проекта
Система рекомендации постов в социальной сети на основе...

## Структура проекта

social-media-recommender/
├── data/ # Исходные данные (не включены в git)
├── notebooks/ # Jupyter-ноутбуки с EDA и экспериментами
├── src/ # Исходный код
│ ├── api/ # FastAPI приложение
│ ├── modeling/ # Обучение и оценка моделей
│ ├── preprocessing/ # Обработка данных и feature engineering
│ └── utils/ # Вспомогательные функции
├── tests/ # Тесты
├── config/ # Конфигурационные файлы
├── docs/ # Документация
├── README.md # Этот файл
├── .gitignore # Игнорируемые файлы
└── requirements.txt # Зависимости