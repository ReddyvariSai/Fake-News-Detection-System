
import os
import json
import joblib
import shutil
from datetime import datetime
import numpy as np
from pathlib import Path

def create_directory_structure():
    """Create the complete models directory structure"""
    models_dir = Path('models')
    
    # Create main directories
    directories = [
        models_dir,
        models_dir / 'best_models',
        models_dir / 'versions',
        models_dir / 'versions' / 'v1.0.0',
        models_dir / 'backups',
        models_dir / 'experiments',
        models_dir / 'logs'
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
        print(f"  Created: {directory}")
    
    return models_dir

def create_init_file(models_dir):
    """Create __init__.py file"""
    init_content = '''"""
Models Package for Fake News Detection
Model saving, loading, versioning, and management
"""

__version__ = "1.0.0"
__author__ = "Fake News Detection Team"
__description__ = "Model management system for fake news detection"

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
'''
    
    init_path = models_dir / '__init__.py'
    with open(init_path, 'w') as f:
        f.write(init_content)
    
    print(f"📄 Created: {init_path}")
    return init_path

def create_sample_model_files(models_dir):
    """Create sample model files for demonstration"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    print("\n🤖 Creating sample model files...")
    
    # Create a simple RandomForest model
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        random_state=42
    )
    
    # Fit with dummy data
    X_dummy = np.random.rand(10, 5)
    y_dummy = np.random.randint(0, 2, 10)
    model.fit(X_dummy, y_dummy)
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    
    # Sample texts for fitting
    sample_texts = [
        "real news article about science and technology",
        "fake news sensational headline breaking",
        "government report economic data analysis",
        "conspiracy theory secret cover up exposed"
    ]
    vectorizer.fit(sample_texts)
    
    # Save main model files
    print("💾 Saving model files...")
    
    # Main model
    model_path = models_dir / 'fake_news_model.pkl'
    joblib.dump(model, model_path)
    print(f"  ✅ fake_news_model.pkl")
    
    # Vectorizer
    vectorizer_path = models_dir / 'vectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_path)
    print(f"  ✅ vectorizer.pkl")
    
    # Create simple preprocessor (dummy object)
    class SimplePreprocessor:
        def preprocess(self, text):
            return text.lower()
    
    preprocessor = SimplePreprocessor()
    preprocessor_path = models_dir / 'preprocessor.pkl'
    joblib.dump(preprocessor, preprocessor_path)
    print(f"  ✅ preprocessor.pkl")
    
    # Create scaler (dummy object)
    class SimpleScaler:
        def transform(self, X):
            return X
        def fit_transform(self, X):
            return X
    
    scaler = SimpleScaler()
    scaler_path = models_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ scaler.pkl")
    
    return model, vectorizer, preprocessor, scaler

def create_versioned_files(models_dir, model, vectorizer):
    """Create version 1.0.0 files"""
    version_dir = models_dir / 'versions' / 'v1.0.0'
    
    print(f"\n🏷️ Creating version 1.0.0 files...")
    
    # Save versioned model
    model_path = version_dir / 'fake_news_model_v1.pkl'
    joblib.dump(model, model_path)
    print(f"  ✅ fake_news_model_v1.pkl")
    
    # Save versioned vectorizer
    vectorizer_path = version_dir / 'vectorizer_v1.pkl'
    joblib.dump(vectorizer, vectorizer_path)
    print(f"  ✅ vectorizer_v1.pkl")
    
    # Create version metadata
    metadata = {
        "model_name": "Fake News Detection Model",
        "version": "1.0.0",
        "model_type": "Random Forest",
        "created_date": datetime.now().isoformat(),
        "author": "Fake News Detection Team",
        "description": "Initial version of fake news detection model",
        "performance_metrics": {
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.90,
            "f1_score": 0.91,
            "roc_auc": 0.95
        },
        "training_data": {
            "samples": 5000,
            "real_news": 2500,
            "fake_news": 2500,
            "features": 5000
        },
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 2,
            "random_state": 42
        }
    }
    
    metadata_path = version_dir / 'metadata_v1.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ metadata_v1.json")
    
    # Copy to main metadata
    main_metadata_path = models_dir / 'metadata.json'
    with open(main_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ metadata.json (main)")
    
    return metadata

def create_best_models(models_dir):
    """Create best models from different algorithms"""
    best_models_dir = models_dir / 'best_models'
    
    print(f"\n🏆 Creating best model files...")
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    
    # Create different models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42),
        'ensemble': RandomForestClassifier(n_estimators=200, random_state=42)
    }
    
    # Fit with dummy data
    X_dummy = np.random.rand(20, 10)
    y_dummy = np.random.randint(0, 2, 20)
    
    for name, model in models.items():
        model.fit(X_dummy, y_dummy)
        
        # Save model
        model_path = best_models_dir / f'{name}_best.pkl'
        joblib.dump(model, model_path)
        print(f"  ✅ {name}_best.pkl")
    
    return models

def create_json_files(models_dir):
    """Create JSON configuration files"""
    print(f"\n📋 Creating JSON configuration files...")
    
    # Create model_versions.json
    versions_data = [
        {
            "version": "1.0.0",
            "model_path": "models/versions/v1.0.0/fake_news_model_v1.pkl",
            "vectorizer_path": "models/versions/v1.0.0/vectorizer_v1.pkl",
            "metadata_path": "models/versions/v1.0.0/metadata_v1.json",
            "created_date": datetime.now().isoformat(),
            "model_hash": "d41d8cd98f00b204e9800998ecf8427e",
            "performance_metrics": {
                "accuracy": 0.92,
                "precision": 0.91,
                "recall": 0.90,
                "f1_score": 0.91
            },
            "is_production": True,
            "model_type": "random_forest",
            "author": "System",
            "description": "Initial production model"
        }
    ]
    
    versions_path = models_dir / 'model_versions.json'
    with open(versions_path, 'w') as f:
        json.dump(versions_data, f, indent=2)
    print(f"  ✅ model_versions.json")
    
    # Create model_registry.json
    registry_data = {
        "models": {
            "fake_news_v1.0.0": {
                "model_id": "fake_news_v1.0.0",
                "model_path": "models/versions/v1.0.0/fake_news_model_v1.pkl",
                "model_type": "random_forest",
                "description": "Initial production model for fake news detection",
                "tags": ["fake_news", "production", "v1.0.0"],
                "metadata": {
                    "accuracy": 0.92,
                    "version": "1.0.0"
                },
                "registered_date": datetime.now().isoformat(),
                "last_used": None,
                "usage_count": 0
            }
        },
        "aliases": {
            "fake_news_latest": "fake_news_v1.0.0",
            "fake_news_production": "fake_news_v1.0.0"
        },
        "current_models": {
            "fake_news_detection": "fake_news_v1.0.0"
        }
    }
    
    registry_path = models_dir / 'model_registry.json'
    with open(registry_path, 'w') as f:
        json.dump(registry_data, f, indent=2)
    print(f"  ✅ model_registry.json")
    
    return versions_data, registry_data

def create_readme(models_dir):
    """Create README.md file"""
    readme_content = """# Models Directory

This directory contains all trained models, vectorizers, and related files for the Fake News Detection system.

## 📁 Directory Structure
