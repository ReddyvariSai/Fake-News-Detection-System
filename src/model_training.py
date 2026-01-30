"""
Model Training Module
Train and evaluate multiple machine learning models for fake news detection
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Import scikit-learn models
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Import scikit-learn utilities
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, 
    StratifiedKFold, learning_curve, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import json

class ModelTrainer:
    """Main model trainer for fake news detection"""
    
    def __init__(self, random_state: int = 42, use_gpu: bool = False):
        """
        Initialize model trainer
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        use_gpu : bool
            Whether to use GPU for training (if available)
        """
        self.random_state = random_state
        self.use_gpu = use_gpu
        
        # Initialize models
        self.models = {}
        self.best_models = {}
        self.cv_results = {}
        self.training_times = {}
        
        # Default models configuration
        self._initialize_models()
        
        # Performance history
        self.performance_history = {}
        
        print(f"Model Trainer initialized with random_state={random_state}")
    
    def _initialize_models(self):
        """Initialize default models"""
        # Logistic Regression
        self.models['logistic_regression'] = {
            'model': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear',
                class_weight='balanced'
            ),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        # Random Forest
        self.models['random_forest'] = {
            'model': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                class_weight='balanced',
                n_jobs=-1
            ),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        }
        
        # XGBoost
        xgb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': self.random_state,
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        if self.use_gpu:
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['gpu_id'] = 0
        
        self.models['xgboost'] = {
            'model': XGBClassifier(**xgb_params),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        }
        
        # Support Vector Machine
        self.models['svm'] = {
            'model': SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            ),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
        
        # Neural Network (MLP)
        self.models['neural_network'] = {
            'model': MLPClassifier(
                random_state=self.random_state,
                max_iter=1000,
                early_stopping=True
            ),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        # LightGBM
        lgbm_params = {
            'random_state': self.random_state,
            'n_estimators': 100,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        if self.use_gpu:
            lgbm_params['device'] = 'gpu'
        
        self.models['lightgbm'] = {
            'model': LGBMClassifier(**lgbm_params),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 63, 127],
                'subsample': [0.6, 0.8, 1.0]
            }
        }
        
        # Naive Bayes
        self.models['naive_bayes'] = {
            'model': MultinomialNB(),
            'params': {
                'alpha': [0.01, 0.1, 1, 10, 100]
            }
        }
    
    def add_custom_model(self, name: str, model, params: Optional[Dict] = None):
        """
        Add a custom model to the trainer
        
        Parameters:
        -----------
        name : str
            Name of the model
        model : sklearn estimator
            Model instance
        params : dict, optional
            Hyperparameters for grid search
        """
        self.models[name] = {
            'model': model,
            'params': params or {}
        }
        print(f"Added custom model: {name}")
    
    def train_single_model(self, X_train, y_train, model_name: str, 
                          use_grid_search: bool = True, cv_folds: int = 5,
                          scoring: str = 'f1', n_jobs: int = -1):
        """
        Train a single model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        model_name : str
            Name of the model to train
        use_grid_search : bool
            Whether to perform hyperparameter tuning
        cv_folds : int
            Number of cross-validation folds
        scoring : str
            Scoring metric for grid search
        n_jobs : int
            Number of parallel jobs
            
        Returns:
        --------
        tuple : (trained_model, training_time, cv_score)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        print(f"\n{'='*60}")
        print(f"Training {model_name.replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        model_info = self.models[model_name]
        model = model_info['model']
        params = model_info['params']
        
        start_time = time.time()
        
        if use_grid_search and params:
            print("Performing hyperparameter tuning...")
            
            # Setup grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                  random_state=self.random_state),
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1,
                refit=True
            )
            
            # Train with grid search
            grid_search.fit(X_train, y_train)
            
            # Store best model
            trained_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
            
            print(f"Best parameters: {best_params}")
            print(f"Best CV {scoring}: {cv_score:.4f}")
            
        else:
            # Train without hyperparameter tuning
            print("Training with default parameters...")
            model.fit(X_train, y_train)
            trained_model = model
            
            # Get cross-validation score
            cv_scores = cross_val_score(
                trained_model, X_train, y_train,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                  random_state=self.random_state),
                scoring=scoring,
                n_jobs=n_jobs
            )
            cv_score = cv_scores.mean()
            print(f"CV {scoring}: {cv_score:.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Store training time
        training_time = time.time() - start_time
        self.training_times[model_name] = training_time
        
        # Store model
        self.best_models[model_name] = trained_model
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return trained_model, training_time, cv_score
    
    def train_all_models(self, X_train, y_train, model_list: Optional[List[str]] = None,
                        use_grid_search: bool = True, cv_folds: int = 5,
                        scoring: str = 'f1', n_jobs: int = -1):
        """
        Train all models or a subset of models
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        model_list : list, optional
            List of model names to train. If None, train all models.
        use_grid_search : bool
            Whether to perform hyperparameter tuning
        cv_folds : int
            Number of cross-validation folds
        scoring : str
            Scoring metric for grid search
        n_jobs : int
            Number of parallel jobs
            
        Returns:
        --------
        dict : Trained models
        """
        if model_list is None:
            model_list = list(self.models.keys())
        
        trained_models = {}
        
        for model_name in model_list:
            try:
                model, train_time, cv_score = self.train_single_model(
                    X_train, y_train, model_name, use_grid_search, 
                    cv_folds, scoring, n_jobs
                )
                trained_models[model_name] = model
                
                # Store CV results
                self.cv_results[model_name] = {
                    'cv_score': cv_score,
                    'training_time': train_time
                }
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Successfully trained {len(trained_models)} models")
        
        return trained_models
    
    def evaluate_models(self, X_test, y_test, model_list: Optional[List[str]] = None):
        """
        Evaluate trained models on test data
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        model_list : list, optional
            List of model names to evaluate. If None, evaluate all trained models.
            
        Returns:
        --------
        pandas.DataFrame : Evaluation metrics for all models
        """
        if not self.best_models:
            raise ValueError("No models trained yet. Call train_all_models first.")
        
        if model_list is None:
            model_list = list(self.best_models.keys())
        
        results = []
        
        for model_name in model_list:
            if model_name not in self.best_models:
                print(f"Model '{model_name}' not found in trained models. Skipping...")
                continue
            
            model = self.best_models[model_name]
            
            print(f"\nEvaluating {model_name.replace('_', ' ').title()}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            metrics = {
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'training_time': self.training_times.get(model_name, 0)
            }
            
            # Add ROC-AUC if probabilities are available
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['true_negative'] = cm[0, 0]
            metrics['false_positive'] = cm[0, 1]
            metrics['false_negative'] = cm[1, 0]
            metrics['true_positive'] = cm[1, 1]
            
            results.append(metrics)
            
            # Print classification report
            print(classification_report(y_test, y_pred, 
                                       target_names=['Real', 'Fake']))
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by F1-score
        results_df = results_df.sort_values('f1_score', ascending=False).reset_index(drop=True)
        
        # Store results
        self.performance_history['test_results'] = results_df
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string())
        
        return results_df
    
    def get_best_model(self, metric: str = 'f1_score'):
        """
        Get the best performing model based on a metric
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison
            
        Returns:
        --------
        tuple : (best_model_name, best_model, metrics_dict)
        """
        if 'test_results' not in self.performance_history:
            raise ValueError("No evaluation results available. Call evaluate_models first.")
        
        results_df = self.performance_history['test_results']
        
        if metric not in results_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results. Available: {list(results_df.columns)}")
        
        # Find best model
        best_idx = results_df[metric].idxmax()
        best_row = results_df.loc[best_idx]
        
        best_model_name = best_row['model']
        best_model = self.best_models[best_model_name]
        
        # Get all metrics for the best model
        best_metrics = best_row.to_dict()
        
        print(f"\nBest Model: {best_model_name.replace('_', ' ').title()}")
        print(f"Best {metric}: {best_metrics[metric]:.4f}")
        
        return best_model_name, best_model, best_metrics
    
    def create_ensemble_model(self, models_to_ensemble: List[str], 
                            ensemble_method: str = 'voting',
                            voting_type: str = 'soft'):
        """
        Create an ensemble of models
        
        Parameters:
        -----------
        models_to_ensemble : list
            List of model names to include in ensemble
        ensemble_method : str
            'voting', 'stacking', or 'blending'
        voting_type : str
            'hard' or 'soft' for voting classifier
            
        Returns:
        --------
        sklearn estimator : Ensemble model
        """
        # Check if models are trained
        for model_name in models_to_ensemble:
            if model_name not in self.best_models:
                raise ValueError(f"Model '{model_name}' not trained yet")
        
        print(f"Creating {ensemble_method} ensemble with models: {models_to_ensemble}")
        
        if ensemble_method == 'voting':
            # Create voting classifier
            estimators = [(name, self.best_models[name]) for name in models_to_ensemble]
            ensemble = VotingClassifier(
                estimators=estimators,
                voting=voting_type,
                n_jobs=-1
            )
            
        elif ensemble_method == 'stacking':
            # Create stacking classifier
            estimators = [(name, self.best_models[name]) for name in models_to_ensemble]
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=5,
                n_jobs=-1
            )
            
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        # Store ensemble
        self.best_models['ensemble'] = ensemble
        self.models['ensemble'] = {'model': ensemble, 'params': {}}
        
        return ensemble
    
    def train_ensemble(self, X_train, y_train, X_test, y_test, 
                      models_to_ensemble: List[str], ensemble_method: str = 'voting'):
        """
        Train and evaluate an ensemble model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        models_to_ensemble : list
            List of model names to include
        ensemble_method : str
            Ensemble method ('voting' or 'stacking')
            
        Returns:
        --------
        sklearn estimator : Trained ensemble model
        """
        # Create ensemble
        ensemble = self.create_ensemble_model(models_to_ensemble, ensemble_method)
        
        print(f"Training ensemble model...")
        start_time = time.time()
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"Ensemble training completed in {training_time:.2f} seconds")
        
        # Evaluate ensemble
        print(f"Evaluating ensemble...")
        y_pred = ensemble.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        print(f"Ensemble Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return ensemble
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save
        filepath : str
            Path to save the model
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        model = self.best_models[model_name]
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def save_all_models(self, directory: str = 'models/'):
        """
        Save all trained models to disk
        
        Parameters:
        -----------
        directory : str
            Directory to save models
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.best_models.items():
            filepath = os.path.join(directory, f"{model_name}_model.pkl")
            joblib.dump(model, filepath)
            print(f"Saved {model_name} to {filepath}")
        
        # Save performance results
        if self.performance_history:
            results_path = os.path.join(directory, 'performance_results.json')
            with open(results_path, 'w') as f:
                # Convert DataFrames to dict for JSON serialization
                json_data = {}
                for key, value in self.performance_history.items():
                    if isinstance(value, pd.DataFrame):
                        json_data[key] = value.to_dict(orient='records')
                    else:
                        json_data[key] = value
                
                json.dump(json_data, f, indent=2)
            print(f"Performance results saved to {results_path}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the model file
            
        Returns:
        --------
        sklearn estimator : Loaded model
        """
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    
    def plot_model_comparison(self, results_df: pd.DataFrame = None, 
                            metrics: List[str] = None, save_path: str = None):
        """
        Plot model comparison
        
        Parameters:
        -----------
        results_df : pandas.DataFrame, optional
            Results dataframe from evaluate_models
        metrics : list, optional
            List of metrics to plot
        save_path : str, optional
            Path to save the plot
        """
        if results_df is None:
            if 'test_results' not in self.performance_history:
                raise ValueError("No evaluation results available")
            results_df = self.performance_history['test_results']
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if not available_metrics:
            raise ValueError("No available metrics to plot")
        
        # Set up plot
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Sort by metric
            sorted_df = results_df.sort_values(metric, ascending=False)
            
            # Create bar plot
            bars = ax.barh(range(len(sorted_df)), sorted_df[metric], 
                          color=plt.cm.Set3(np.linspace(0, 1, len(sorted_df))))
            
            ax.set_yticks(range(len(sorted_df)))
            ax.set_yticklabels([name.replace('_', ' ').title() for name in sorted_df['model']])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.invert_yaxis()  # Highest on top
            
            # Add value labels
            for bar, value in zip(bars, sorted_df[metric]):
                ax.text(value, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()

# Usage example
if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, 
                              n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Train all models
    trained_models = trainer.train_all_models(
        X_train, y_train,
        model_list=['logistic_regression', 'random_forest', 'xgboost'],
        use_grid_search=False
    )
    
    # Evaluate models
    results_df = trainer.evaluate_models(X_test, y_test)
    
    # Get best model
    best_name, best_model, metrics = trainer.get_best_model()
    
    # Save best model
    trainer.save_model(best_name, f"{best_name}_best_model.pkl")
