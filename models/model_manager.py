"""
Model Manager Module
Handles model saving, loading, versioning, and metadata management
"""

import os
import json
import joblib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import hashlib
import shutil
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_name: str
    model_type: str
    model_version: str
    created_date: str
    author: str
    description: str
    performance_metrics: Dict[str, float]
    training_data_info: Dict[str, Any]
    feature_info: Dict[str, Any]
    dependencies: Dict[str, str]
    hyperparameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save metadata to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ModelMetadata':
        """Load metadata from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

@dataclass
class ModelVersion:
    """Model version information"""
    version: str
    model_path: str
    vectorizer_path: str
    metadata_path: str
    created_date: str
    performance_score: float
    is_production: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class ModelManager:
    """Main model manager for saving, loading, and versioning models"""
    
    def __init__(self, models_dir: str = 'models/'):
        """
        Initialize model manager
        
        Parameters:
        -----------
        models_dir : str
            Directory to store models
        """
        self.models_dir = models_dir
        self.versions_file = os.path.join(models_dir, 'model_versions.json')
        self.registry_file = os.path.join(models_dir, 'model_registry.json')
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Load existing versions and registry
        self.versions = self._load_versions()
        self.registry = self._load_registry()
        
        # Current model paths
        self.current_model_path = os.path.join(models_dir, 'fake_news_model.pkl')
        self.current_vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
        self.current_scaler_path = os.path.join(models_dir, 'scaler.pkl')
        self.current_preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
        self.current_metadata_path = os.path.join(models_dir, 'metadata.json')
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.models_dir,
            os.path.join(self.models_dir, 'best_models'),
            os.path.join(self.models_dir, 'versions'),
            os.path.join(self.models_dir, 'backups'),
            os.path.join(self.models_dir, 'experiments'),
            os.path.join(self.models_dir, 'logs')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_versions(self) -> List[Dict[str, Any]]:
        """Load model versions from file"""
        if os.path.exists(self.versions_file):
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return []
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'models': {}, 'current_version': None}
    
    def _save_versions(self):
        """Save model versions to file"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def _save_registry(self):
        """Save model registry to file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def calculate_model_hash(self, model_path: str) -> str:
        """
        Calculate MD5 hash of model file
        
        Parameters:
        -----------
        model_path : str
            Path to model file
            
        Returns:
        --------
        str : MD5 hash
        """
        if not os.path.exists(model_path):
            return ""
        
        with open(model_path, 'rb') as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
        
        return file_hash.hexdigest()
    
    def save_model(self, model, model_name: str = 'fake_news_model',
                  model_type: str = 'random_forest', version: str = None,
                  performance_metrics: Optional[Dict] = None,
                  training_data_info: Optional[Dict] = None,
                  feature_info: Optional[Dict] = None,
                  hyperparameters: Optional[Dict] = None,
                  author: str = "System", description: str = "",
                  is_production: bool = False, dependencies: Optional[Dict] = None):
        """
        Save a model with metadata and versioning
        
        Parameters:
        -----------
        model : sklearn model
            Trained model to save
        model_name : str
            Name of the model
        model_type : str
            Type of model (e.g., 'random_forest', 'xgboost')
        version : str, optional
            Version string (e.g., '1.0.0'). If None, auto-increment.
        performance_metrics : dict, optional
            Model performance metrics
        training_data_info : dict, optional
            Information about training data
        feature_info : dict, optional
            Information about features
        hyperparameters : dict, optional
            Model hyperparameters
        author : str
            Author name
        description : str
            Model description
        is_production : bool
            Whether this is the production model
        dependencies : dict, optional
            Package dependencies
            
        Returns:
        --------
        str : Version string
        """
        if version is None:
            # Auto-generate version
            version = self._generate_version()
        
        # Create version directory
        version_dir = os.path.join(self.models_dir, 'versions', f'v{version}')
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(version_dir, f'{model_name}_v{version}.pkl')
        joblib.dump(model, model_path)
        
        # Calculate model hash
        model_hash = self.calculate_model_hash(model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            model_type=model_type,
            model_version=version,
            created_date=datetime.now().isoformat(),
            author=author,
            description=description,
            performance_metrics=performance_metrics or {},
            training_data_info=training_data_info or {},
            feature_info=feature_info or {},
            dependencies=dependencies or {
                'scikit-learn': '1.0+',
                'numpy': '1.20+',
                'pandas': '1.3+'
            },
            hyperparameters=hyperparameters or {}
        )
        
        # Save metadata
        metadata_path = os.path.join(version_dir, f'metadata_v{version}.json')
        metadata.to_json(metadata_path)
        
        # Add to versions
        version_info = {
            'version': version,
            'model_path': model_path,
            'metadata_path': metadata_path,
            'created_date': metadata.created_date,
            'model_hash': model_hash,
            'performance_metrics': performance_metrics,
            'is_production': is_production,
            'model_type': model_type,
            'author': author
        }
        
        self.versions.append(version_info)
        self._save_versions()
        
        # Update registry
        self.registry['models'][version] = version_info
        if is_production:
            self.registry['current_version'] = version
        
        self._save_registry()
        
        # Copy to current model if it's production
        if is_production:
            self._set_production_model(version)
        
        print(f"Model saved as version {version} at {model_path}")
        print(f"Model hash: {model_hash}")
        
        return version
    
    def _generate_version(self) -> str:
        """Generate new version number"""
        if not self.versions:
            return "1.0.0"
        
        # Get all version numbers
        versions = [v['version'] for v in self.versions]
        
        # Parse version numbers
        version_numbers = []
        for v in versions:
            try:
                parts = v.split('.')
                major = int(parts[0]) if len(parts) > 0 else 0
                minor = int(parts[1]) if len(parts) > 1 else 0
                patch = int(parts[2]) if len(parts) > 2 else 0
                version_numbers.append((major, minor, patch))
            except:
                continue
        
        if not version_numbers:
            return "1.0.0"
        
        # Get latest version
        latest = max(version_numbers)
        
        # Increment patch version
        new_version = f"{latest[0]}.{latest[1]}.{latest[2] + 1}"
        
        return new_version
    
    def _set_production_model(self, version: str):
        """
        Set a model version as production
        
        Parameters:
        -----------
        version : str
            Version to set as production
        """
        # Find version info
        version_info = None
        for v in self.versions:
            if v['version'] == version:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Version {version} not found")
        
        # Load model and metadata
        model = joblib.load(version_info['model_path'])
        
        # Copy to current model location
        shutil.copy2(version_info['model_path'], self.current_model_path)
        
        # Update registry
        for v in self.versions:
            v['is_production'] = (v['version'] == version)
        
        self.registry['current_version'] = version
        self._save_versions()
        self._save_registry()
        
        print(f"Version {version} set as production model")
    
    def load_model(self, version: Optional[str] = None, 
                  load_metadata: bool = True) -> Tuple[Any, Optional[Dict]]:
        """
        Load a model by version
        
        Parameters:
        -----------
        version : str, optional
            Version to load. If None, load production model.
        load_metadata : bool
            Whether to load metadata
            
        Returns:
        --------
        tuple : (model, metadata)
        """
        if version is None:
            version = self.registry.get('current_version')
            if not version:
                raise ValueError("No production model set")
        
        # Find version info
        version_info = None
        for v in self.versions:
            if v['version'] == version:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Version {version} not found")
        
        # Load model
        model_path = version_info['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        
        # Load metadata if requested
        metadata = None
        if load_metadata and 'metadata_path' in version_info:
            metadata_path = version_info['metadata_path']
            if os.path.exists(metadata_path):
                metadata = ModelMetadata.from_json(metadata_path)
        
        print(f"Loaded model version {version} from {model_path}")
        
        return model, metadata.to_dict() if metadata else None
    
    def load_production_model(self) -> Tuple[Any, Optional[Dict]]:
        """Load the current production model"""
        return self.load_model()
    
    def get_model_versions(self) -> List[Dict[str, Any]]:
        """Get list of all model versions"""
        return self.versions
    
    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific version"""
        for v in self.versions:
            if v['version'] == version:
                return v
        return None
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Parameters:
        -----------
        version1 : str
            First version
        version2 : str
            Second version
            
        Returns:
        --------
        dict : Comparison results
        """
        info1 = self.get_version_info(version1)
        info2 = self.get_version_info(version2)
        
        if not info1 or not info2:
            raise ValueError("One or both versions not found")
        
        comparison = {
            'versions': [version1, version2],
            'creation_dates': [info1['created_date'], info2['created_date']],
            'model_types': [info1.get('model_type', 'unknown'), 
                          info2.get('model_type', 'unknown')],
            'is_production': [info1.get('is_production', False),
                            info2.get('is_production', False)]
        }
        
        # Compare performance metrics if available
        if 'performance_metrics' in info1 and 'performance_metrics' in info2:
            metrics1 = info1['performance_metrics']
            metrics2 = info2['performance_metrics']
            
            common_metrics = set(metrics1.keys()) & set(metrics2.keys())
            metric_comparison = {}
            
            for metric in common_metrics:
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                diff = val2 - val1
                improvement = "better" if diff > 0 else "worse" if diff < 0 else "same"
                
                metric_comparison[metric] = {
                    'version1': val1,
                    'version2': val2,
                    'difference': diff,
                    'improvement': improvement
                }
            
            comparison['metric_comparison'] = metric_comparison
        
        return comparison
    
    def delete_version(self, version: str, confirm: bool = True):
        """
        Delete a model version
        
        Parameters:
        -----------
        version : str
            Version to delete
        confirm : bool
            Whether to ask for confirmation
        """
        # Find version info
        version_info = self.get_version_info(version)
        if not version_info:
            raise ValueError(f"Version {version} not found")
        
        if version_info.get('is_production', False):
            raise ValueError("Cannot delete production version. Set another version as production first.")
        
        if confirm:
            response = input(f"Are you sure you want to delete version {version}? (y/n): ")
            if response.lower() != 'y':
                print("Deletion cancelled")
                return
        
        # Remove from lists
        self.versions = [v for v in self.versions if v['version'] != version]
        
        if version in self.registry['models']:
            del self.registry['models'][version]
        
        # Update current version if needed
        if self.registry['current_version'] == version:
            self.registry['current_version'] = None
        
        # Delete files
        model_path = version_info['model_path']
        metadata_path = version_info.get('metadata_path')
        
        if os.path.exists(model_path):
            os.remove(model_path)
        
        if metadata_path and os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Remove version directory if empty
        version_dir = os.path.dirname(model_path)
        if os.path.exists(version_dir) and not os.listdir(version_dir):
            os.rmdir(version_dir)
        
        # Save updated data
        self._save_versions()
        self._save_registry()
        
        print(f"Version {version} deleted successfully")
    
    def export_model(self, version: str, export_dir: str):
        """
        Export a model version to a directory
        
        Parameters:
        -----------
        version : str
            Version to export
        export_dir : str
            Directory to export to
        """
        version_info = self.get_version_info(version)
        if not version_info:
            raise ValueError(f"Version {version} not found")
        
        os.makedirs(export_dir, exist_ok=True)
        
        # Copy model file
        model_src = version_info['model_path']
        model_dst = os.path.join(export_dir, f'model_v{version}.pkl')
        shutil.copy2(model_src, model_dst)
        
        # Copy metadata if exists
        metadata_src = version_info.get('metadata_path')
        if metadata_src and os.path.exists(metadata_src):
            metadata_dst = os.path.join(export_dir, f'metadata_v{version}.json')
            shutil.copy2(metadata_src, metadata_dst)
        
        # Create export info file
        export_info = {
            'export_date': datetime.now().isoformat(),
            'version': version,
            'model_path': model_dst,
            'metadata_path': metadata_dst if metadata_src else None,
            'model_info': version_info
        }
        
        info_path = os.path.join(export_dir, f'export_info_v{version}.json')
        with open(info_path, 'w') as f:
            json.dump(export_info, f, indent=2)
        
        print(f"Version {version} exported to {export_dir}")
    
    def save_complete_pipeline(self, model, vectorizer, scaler=None, 
                              preprocessor=None, version=None, **kwargs):
        """
        Save complete ML pipeline
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        vectorizer : sklearn vectorizer
            Feature vectorizer
        scaler : sklearn scaler, optional
            Feature scaler
        preprocessor : object, optional
            Text preprocessor
        version : str, optional
            Version string
        **kwargs : dict
            Additional arguments for save_model
        """
        if version is None:
            version = self._generate_version()
        
        # Create version directory
        version_dir = os.path.join(self.models_dir, 'versions', f'v{version}')
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(version_dir, f'model_v{version}.pkl')
        joblib.dump(model, model_path)
        
        # Save vectorizer
        vectorizer_path = os.path.join(version_dir, f'vectorizer_v{version}.pkl')
        joblib.dump(vectorizer, vectorizer_path)
        
        # Save scaler if provided
        scaler_path = None
        if scaler:
            scaler_path = os.path.join(version_dir, f'scaler_v{version}.pkl')
            joblib.dump(scaler, scaler_path)
        
        # Save preprocessor if provided
        preprocessor_path = None
        if preprocessor:
            preprocessor_path = os.path.join(version_dir, f'preprocessor_v{version}.pkl')
            joblib.dump(preprocessor, preprocessor_path)
        
        # Update feature info in kwargs
        if 'feature_info' not in kwargs:
            kwargs['feature_info'] = {}
        
        kwargs['feature_info'].update({
            'vectorizer_type': type(vectorizer).__name__,
            'scaler_used': scaler is not None,
            'preprocessor_used': preprocessor is not None,
            'feature_count': vectorizer.get_feature_names_out().shape[0] 
            if hasattr(vectorizer, 'get_feature_names_out') else 'unknown'
        })
        
        # Save with metadata
        self.save_model(model, version=version, **kwargs)
        
        # Update version info with pipeline paths
        for v in self.versions:
            if v['version'] == version:
                v['vectorizer_path'] = vectorizer_path
                v['scaler_path'] = scaler_path
                v['preprocessor_path'] = preprocessor_path
                break
        
        self._save_versions()
        
        # Copy to current locations if production
        if kwargs.get('is_production', False):
            shutil.copy2(model_path, self.current_model_path)
            shutil.copy2(vectorizer_path, self.current_vectorizer_path)
            
            if scaler_path:
                shutil.copy2(scaler_path, self.current_scaler_path)
            
            if preprocessor_path:
                shutil.copy2(preprocessor_path, self.current_preprocessor_path)
        
        print(f"Complete pipeline saved as version {version}")
    
    def load_complete_pipeline(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load complete ML pipeline
        
        Parameters:
        -----------
        version : str, optional
            Version to load
            
        Returns:
        --------
        dict : Loaded pipeline components
        """
        model, metadata = self.load_model(version)
        
        # Find version info
        version_info = None
        for v in self.versions:
            if v['version'] == (version or self.registry['current_version']):
                version_info = v
                break
        
        if not version_info:
            raise ValueError("Version info not found")
        
        # Load vectorizer
        vectorizer = None
        if 'vectorizer_path' in version_info and os.path.exists(version_info['vectorizer_path']):
            vectorizer = joblib.load(version_info['vectorizer_path'])
        
        # Load scaler
        scaler = None
        if 'scaler_path' in version_info and os.path.exists(version_info['scaler_path']):
            scaler = joblib.load(version_info['scaler_path'])
        
        # Load preprocessor
        preprocessor = None
        if 'preprocessor_path' in version_info and os.path.exists(version_info['preprocessor_path']):
            preprocessor = joblib.load(version_info['preprocessor_path'])
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'scaler': scaler,
            'preprocessor': preprocessor,
            'metadata': metadata,
            'version_info': version_info
        }

# Model registry for tracking multiple models
class ModelRegistry:
    """Registry for managing multiple models"""
    
    def __init__(self, registry_file: str = 'models/model_registry.json'):
        """
        Initialize model registry
        
        Parameters:
        -----------
        registry_file : str
            Path to registry file
        """
        self.registry_file = registry_file
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {
            'models': {},
            'aliases': {},
            'current_models': {}
        }
    
    def _save_registry(self):
        """Save registry to file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_id: str, model_path: str, 
                      model_type: str, description: str = "",
                      tags: Optional[List[str]] = None,
                      metadata: Optional[Dict] = None):
        """
        Register a model in the registry
        
        Parameters:
        -----------
        model_id : str
            Unique identifier for the model
        model_path : str
            Path to model file
        model_type : str
            Type of model
        description : str
            Model description
        tags : list, optional
            List of tags
        metadata : dict, optional
            Additional metadata
        """
        if model_id in self.registry['models']:
            print(f"Model {model_id} already registered. Updating...")
        
        model_info = {
            'model_id': model_id,
            'model_path': model_path,
            'model_type': model_type,
            'description': description,
            'tags': tags or [],
            'metadata': metadata or {},
            'registered_date': datetime.now().isoformat(),
            'last_used': None,
            'usage_count': 0
        }
        
        self.registry['models'][model_id] = model_info
        self._save_registry()
        
        print(f"Model {model_id} registered successfully")
    
    def register_alias(self, alias: str, model_id: str):
        """
        Register an alias for a model
        
        Parameters:
        -----------
        alias : str
            Alias name
        model_id : str
            Model ID to alias
        """
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found in registry")
        
        self.registry['aliases'][alias] = model_id
        self._save_registry()
        
        print(f"Alias '{alias}' registered for model {model_id}")
    
    def set_current_model(self, task: str, model_id: str):
        """
        Set current model for a task
        
        Parameters:
        -----------
        task : str
            Task name (e.g., 'fake_news_detection')
        model_id : str
            Model ID to set as current
        """
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found in registry")
        
        self.registry['current_models'][task] = model_id
        self._save_registry()
        
        print(f"Model {model_id} set as current for task '{task}'")
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model info by ID"""
        return self.registry['models'].get(model_id)
    
    def get_model_by_alias(self, alias: str) -> Optional[Dict[str, Any]]:
        """Get model info by alias"""
        model_id = self.registry['aliases'].get(alias)
        if model_id:
            return self.get_model(model_id)
        return None
    
    def get_current_model(self, task: str) -> Optional[Dict[str, Any]]:
        """Get current model for a task"""
        model_id = self.registry['current_models'].get(task)
        if model_id:
            return self.get_model(model_id)
        return None
    
    def list_models(self, model_type: Optional[str] = None, 
                   tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List registered models
        
        Parameters:
        -----------
        model_type : str, optional
            Filter by model type
        tag : str, optional
            Filter by tag
            
        Returns:
        --------
        list : List of model info dictionaries
        """
        models = list(self.registry['models'].values())
        
        if model_type:
            models = [m for m in models if m['model_type'] == model_type]
        
        if tag:
            models = [m for m in models if tag in m.get('tags', [])]
        
        return models
    
    def update_model_usage(self, model_id: str):
        """Update model usage statistics"""
        if model_id in self.registry['models']:
            self.registry['models'][model_id]['last_used'] = datetime.now().isoformat()
            self.registry['models'][model_id]['usage_count'] = \
                self.registry['models'][model_id].get('usage_count', 0) + 1
            self._save_registry()
    
    def delete_model(self, model_id: str, confirm: bool = True):
        """
        Delete a model from registry
        
        Parameters:
        -----------
        model_id : str
            Model ID to delete
        confirm : bool
            Whether to ask for confirmation
        """
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found in registry")
        
        if confirm:
            response = input(f"Are you sure you want to delete model {model_id}? (y/n): ")
            if response.lower() != 'y':
                print("Deletion cancelled")
                return
        
        # Remove from registry
        del self.registry['models'][model_id]
        
        # Remove aliases
        aliases_to_remove = []
        for alias, mid in self.registry['aliases'].items():
            if mid == model_id:
                aliases_to_remove.append(alias)
        
        for alias in aliases_to_remove:
            del self.registry['aliases'][alias]
        
        # Remove from current models
        tasks_to_remove = []
        for task, mid in self.registry['current_models'].items():
            if mid == model_id:
                tasks_to_remove.append(task)
        
        for task in tasks_to_remove:
            del self.registry['current_models'][task]
        
        self._save_registry()
        
        print(f"Model {model_id} removed from registry")
