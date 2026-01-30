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
        self.use
