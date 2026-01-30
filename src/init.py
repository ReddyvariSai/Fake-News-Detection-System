
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
