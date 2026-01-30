"""
Fake News Detection Package
Machine Learning for Truth Discovery
"""

__version__ = "1.0.0"
__author__ = "Fake News Detection Team"
__description__ = "A comprehensive machine learning system for detecting fake news"

# Import key classes for easier access
from .data_preprocessing import DataPreprocessor, EnhancedDataPreprocessor
from .feature_extraction import FeatureExtractor, FakeNewsFeatureExtractor
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator
from .prediction import NewsPredictor, RealTimePredictor

# Package exports
__all__ = [
    'DataPreprocessor',
    'EnhancedDataPreprocessor',
    'FeatureExtractor',
    'FakeNewsFeatureExtractor',
    'ModelTrainer',
    'ModelEvaluator',
    'NewsPredictor',
    'RealTimePredictor'
]

print(f"Fake News Detection System v{__version__}")
