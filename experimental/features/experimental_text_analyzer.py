"""
NOTE: Этот модуль содержит экспериментальный класс
для извлечения признаков из текста постов.
Класс не используется в текущей версии модели.
Сохранен для возможного будущего использования.
"""
import re
import logging
from typing import Dict, List, Union, Optional, Tuple
from collections import defaultdict
import spacy
from functools import lru_cache
import pandas as pd

# Настройка логирования
logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self, config: Dict = None):
        """
        Инициализирует анализатор текста с конфигурацией из config.yaml
        
        :param config: Конфигурация из раздела feature_engineering.text_analyzer
        """
        # Получаем конфигурацию или используем значения по умолчанию
        self.config = config or self.get_default_config()
        logger.info("Initializing TextAnalyzer with config: %s", self.config)
        
        # Инициализируем NLP модель с кэшированием
        self.nlp = self._init_nlp_model()
        
        # Компилируем паттерны
        self.patterns = self._compile_patterns()
        self.truncation_regexes = self._compile_truncation_patterns()
        
        # Инициализируем стоп-слова
        self.stop_words = set(self.config.get("stop_words", []))
    
    @lru_cache(maxsize=2)
    def _load_spacy_model(self, model_name: str, disable: tuple) -> spacy.Language:
        """Загружает модель spaCy с кэшированием"""
        try:
            logger.debug("Loading spaCy model: %s", model_name)
            return spacy.load(model_name, disable=disable)
        except OSError:
            logger.warning("Model %s not found, downloading...", model_name)
            spacy.cli.download(model_name)
            return spacy.load(model_name, disable=disable)
    
    def _init_nlp_model(self) -> spacy.Language:
        """Инициализирует модель spaCy с учетом конфигурации"""
        model_name = self.config.get("language_model", "en_core_web_sm")
        disable_components = tuple(self.config.get("disable_components", ["parser", "ner"]))
        return self._load_spacy_model(model_name, disable_components)
    
    def _compile_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Компилирует regex-паттерны для извлечения сущностей"""
        compiled = []
        for pattern_config in self.config["patterns"]:
            try:
                # Подготовка флагов для компиляции
                flags = 0
                if pattern_config.get("flags"):
                    for flag in pattern_config["flags"].split("|"):
                        if flag == "VERBOSE": flags |= re.VERBOSE
                        if flag == "IGNORECASE": flags |= re.IGNORECASE
                        if flag == "MULTILINE": flags |= re.MULTILINE
                
                compiled.append((
                    re.compile(pattern_config["pattern"], flags),
                    pattern_config["name"]
                ))
            except re.error as e:
                logger.error("Error compiling pattern '%s': %s", pattern_config["name"], str(e))
            except KeyError as e:
                logger.error("Missing required key in pattern config: %s", str(e))
        
        return compiled
    
    def _compile_truncation_patterns(self) -> List[re.Pattern]:
        """Компилирует паттерны для обнаружения усечённых текстов"""
        compiled = []
        for pattern in self.config["truncation_patterns"]:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.error("Error compiling truncation pattern '%s': %s", pattern, str(e))
        return compiled
    
    @staticmethod
    def get_default_config() -> Dict:
        """Возвращает конфигурацию по умолчанию"""
        return {
            "patterns": [
                {
                    "name": "phones",
                    "pattern": r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,3}\)?[-.\s]?\d{2,3}[-.\s]?\d{2,4}(?:\s*(?:ext|x|доб)[-.]?\d{2,5})?",
                    "flags": "VERBOSE"
                },
                {
                    "name": "dates",
                    "pattern": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
                },
                {
                    "name": "years",
                    "pattern": r"\b(?:19|20)\d{2}s?\b"
                },
                {
                    "name": "mentions",
                    "pattern": r"@\w+"
                },
                {
                    "name": "hashtags",
                    "pattern": r"#\w+"
                },
                {
                    "name": "urls",
                    "pattern": r"https?://\S+"
                },
                {
                    "name": "emails",
                    "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                }
            ],
            "truncation_patterns": [
                r"\.{3}\s*https?://\S+$",
                r"…\s*https?://\S+$",
                r"\bcontinued?\s*https?://\S+$",
                r"\bread more\b.*https?://\S+$",
                r"…\s*https://t\.co/\w+$",
                r"\.{3}\s*https://t\.co/\w+$",
                r"https?://t\.co/\w+$"
            ],
            "language_model": "en_core_web_sm",
            "disable_components": ["parser", "ner"],
            "stop_words": []
        }
    
    def _is_truncated(self, text: str) -> bool:
        """Проверяет, является ли текст усечённым с ссылкой для продолжения"""
        return any(pattern.search(text) for pattern in self.truncation_regexes) or (
            re.search(r'https?://\S+$', text) and len(text) < 100
        )
    
    def _extract_continuation_url(self, text: str) -> Optional[str]:
        """Извлекает URL для продолжения из усечённого текста"""
        match = re.search(r'(https?://\S+)$', text)
        return match.group(1) if match else None
    
    def _expand_decade_year(self, year_str: str) -> List[int]:
        """Преобразует год в формате 1980s в диапазон годов (1981-1989)"""
        if year_str.endswith('s'):
            base_year = int(year_str[:-1])
            return list(range(base_year + 1, base_year + 10))
        return [int(year_str)]
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Извлекает именованные сущности из текста"""
        try:
            doc = self.nlp(text)
            entities = defaultdict(list)
            
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "GPE", "LOC", "ORG"}:
                    entity_type = ent.label_.lower() + "s"
                    entities[entity_type].append(ent.text)
                    
            return dict(entities)
        except Exception as e:
            logger.error("Error extracting entities: %s", str(e))
            return {}
    
    def _calculate_metrics(self, text: str) -> Dict[str, Union[float, int]]:
        """Вычисляет лингвистические метрики текста"""
        if not text.strip():
            return {
                'word_count': 0,
                'avg_word_length': 0.0,
                'paragraph_count': 0,
                'avg_paragraph_length': 0.0,
                'sentence_count': 0,
                'avg_sentence_length': 0.0,
                'is_truncated': False
            }
        
        try:
            # Разделение на абзацы и предложения
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            
            # Токенизация слов
            words = [
                word.lower() for word in re.findall(r'\w+', text.lower()) 
                if word.lower() not in self.stop_words
            ]
            
            # Расчёт метрик
            word_count = len(words)
            avg_word_length = round(sum(len(word) for word in words) / word_count, 2) if words else 0.0
            
            return {
                'word_count': word_count,
                'avg_word_length': avg_word_length,
                'paragraph_count': len(paragraphs),
                'avg_paragraph_length': round(sum(len(p.split()) for p in paragraphs) / len(paragraphs), 2) if paragraphs else 0.0,
                'sentence_count': len(sentences),
                'avg_sentence_length': round(sum(len(s.split()) for s in sentences) / len(sentences), 2) if sentences else 0.0,
                'is_truncated': self._is_truncated(text)
            }
        except Exception as e:
            logger.error("Error calculating text metrics: %s", str(e))
            return {
                'word_count': 0,
                'avg_word_length': 0.0,
                'paragraph_count': 0,
                'avg_paragraph_length': 0.0,
                'sentence_count': 0,
                'avg_sentence_length': 0.0,
                'is_truncated': False
            }
    
    def analyze(self, text: str, include_tokens: bool = True) -> Dict:
        """
        Выполняет комплексный анализ текста
        
        :param text: Текст для анализа
        :param include_tokens: Включать ли информацию о токенах
        :return: Словарь с результатами анализа
        """
        if not text:
            return {}
            
        try:
            # Базовые метрики
            metrics = self._calculate_metrics(text)
            result = {'metrics': metrics}
            
            # Извлечение специальных токенов
            special_tokens = defaultdict(list)
            remaining_text = text
            
            for pattern, token_type in self.patterns:
                for match in pattern.finditer(remaining_text):
                    token = match.group()
                    if token_type == 'years':
                        years = self._expand_decade_year(token)
                        special_tokens[token_type].extend(years)
                    else:
                        special_tokens[token_type].append(token)
                        # Заменяем только первое вхождение для безопасности
                        remaining_text = remaining_text.replace(token, ' ', 1)
            
            result['special_tokens'] = dict(special_tokens)
            
            # Извлечение сущностей
            result['named_entities'] = self._extract_entities(text)
            
            # URL продолжения для усечённых текстов
            if metrics['is_truncated']:
                result['continuation_url'] = self._extract_continuation_url(text)
            
            # Дополнительная информация о токенах
            if include_tokens:
                words = [
                    word for word in re.findall(r'\b\w+\b', remaining_text.lower())
                    if word not in self.stop_words and len(word) > 1
                ]
                
                all_tokens = []
                for token_list in special_tokens.values():
                    all_tokens.extend(token_list if isinstance(token_list, list) else [token_list])
                all_tokens.extend(words)
                
                result.update({
                    'tokens': words,
                    'all_tokens': all_tokens
                })
            return result
        
        except Exception as e:
            logger.exception("Error during text analysis")
            return {}
        
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        include_tokens: bool = True,
        drop_original: bool = False,
        parallel: bool = False,
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        Анализирует текстовую колонку в датафрейме и возвращает новый датафрейм с результатами.

        Параметры:
        ----------
        df : pd.DataFrame
            Входной датафрейм
        text_column : str
            Название колонки с текстом для анализа
        include_tokens : bool
            Включать ли токены в результат
        drop_original : bool
            Удалять ли исходную текстовую колонку
        parallel : bool
            Использовать ли параллельную обработку
        n_jobs : int
            Количество ядер для параллельной обработки (-1 = все ядра)

        Возвращает:
        -----------
        pd.DataFrame
            Датафрейм с добавленными признаками
        """
        if text_column not in df.columns:
            raise ValueError(f"Колонка '{text_column}' не найдена в датафрейме")

        
        # Параллельная обработка при необходимости
        if parallel:
            try:
                from pandarallel import pandarallel
                pandarallel.initialize(nb_workers=n_jobs)
                analysis_results = df[text_column].parallel_apply(
                    lambda x: self.analyze(x, include_tokens=include_tokens)
                )
            except ImportError:
                logger.warning("pandarallel не установлен, используется обычная обработка")
                analysis_results = df[text_column].apply(
                    lambda x: self.analyze(x, include_tokens=include_tokens)
                )
        else:
            analysis_results = df[text_column].apply(
                lambda x: self.analyze(x, include_tokens=include_tokens)
            )

        # Преобразуем результаты в DataFrame
        expanded_data = []
        
        for result in analysis_results:
            row = {}
            
            # Метрики
            if 'metrics' in result:
                for metric, value in result['metrics'].items():
                    row[f'text_{metric}'] = value
            
            # Специальные токены
            if 'special_tokens' in result:
                for token_type, tokens in result['special_tokens'].items():
                    row[f'text_{token_type}_count'] = len(tokens)
                    if tokens:
                        row[f'text_{token_type}'] = tokens
            
            # Именованные сущности
            if 'named_entities' in result:
                for entity_type, entities in result['named_entities'].items():
                    row[f'text_{entity_type}_count'] = len(entities)
                    if entities:
                        row[f'text_{entity_type}'] = entities
            
            # Дополнительные поля
            if 'continuation_url' in result:
                row['text_continuation_url'] = result['continuation_url']
            
            expanded_data.append(row)

        expanded_df = pd.DataFrame(expanded_data)
        
        # Объединяем с исходным датафреймом
        result_df = pd.concat([df.reset_index(drop=True), 
                              expanded_df.reset_index(drop=True)], axis=1)
        
        if drop_original:
            result_df = result_df.drop(columns=[text_column])
        
        return result_df
        