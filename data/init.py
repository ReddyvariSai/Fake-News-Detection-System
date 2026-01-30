"""
Data Management Package for Fake News Detection
"""

from .data_manager import DataManager
from .data_processor import DataProcessor
from .data_validator import DataValidator
from .data_splitter import DataSplitter
from .data_augmenter import DataAugmenter

__version__ = "1.0.0"
__all__ = [
    "DataManager", 
    "DataProcessor", 
    "DataValidator", 
    "DataSplitter", 
    "DataAugmenter"
]
