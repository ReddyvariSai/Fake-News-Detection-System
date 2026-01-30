import os
import json
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Export key classes
from .model_manager import ModelManager, ModelRegistry, ModelMetadata, ModelVersion

# Package exports
__all__ = [
    'ModelManager',
    'ModelRegistry', 
    'ModelMetadata',
    'ModelVersion',
    'load_model',
    'save_model',
    'load_production_model',
    'list_model_versions'
]

# Utility functions
def load_model(model_path: str):
    """
    Load a model from file
    
    Parameters:
    -----------
    model_path : str
        Path to model file
        
    Returns:
    --------
    object : Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)

def save_model(model, model_path: str, compress: int = 3):
    """
    Save a model to file
    
    Parameters:
    -----------
    model : object
        Model to save
    model_path : str
        Path to save model
    compress : int
        Compression level (0-9)
    """
    joblib.dump(model, model_path, compress=compress)
    print(f"Model saved to {model_path}")

def load_production_model(models_dir: str = 'models/'):
    """
    Load the production model
    
    Parameters:
    -----------
    models_dir : str
        Models directory
        
    Returns:
    --------
    tuple : (model, metadata)
    """
    manager = ModelManager(models_dir)
    return manager.load_production_model()

def list_model_versions(models_dir: str = 'models/'):
    """
    List all model versions
    
    Parameters:
    -----------
    models_dir : str
        Models directory
        
    Returns:
    --------
    list : List of version information
    """
    manager = ModelManager(models_dir)
    return manager.get_model_versions()

# Initialize package
print(f"📦 Fake News Detection Model Management v{__version__}")
print(f"📁 Models directory: {os.path.abspath('models/')}")
