"""
Feature Extraction Module
Convert text data into numerical features for machine learning
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from collections import Counter
import re

class FeatureExtractor:
    """Main feature extractor for text data"""
    
    def __init__(self, max_features=5000, method='tfidf', 
                 ngram_range=(1, 2), use_svd=False, n_components=100):
        """
        Initialize feature extractor
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features
        method : str
            Feature extraction method ('tfidf', 'count', 'binary')
        ngram_range : tuple
            Range of n-grams to extract
        use_svd : bool
            Whether to apply dimensionality reduction
        n_components : int
            Number of components for SVD
        """
        self.max_features = max_features
        self.method = method
        self.ngram_range = ngram_range
        self.use_svd = use_svd
        self.n_components = n_components
        
        # Initialize vectorizers
        self._initialize_vectorizers()
        
        # Initialize scalers
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # Store feature names
        self.feature_names = None
        self.is_fitted = False
    
    def _initialize_vectorizers(self):
        """Initialize text vectorizers based on method"""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                min_df=2,
                max_df=0.95,
                binary=False
            )
        elif self.method == 'binary':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                min_df=2,
                max_df=0.95,
                binary=True
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'tfidf', 'count', or 'binary'")
        
        # Initialize SVD for dimensionality reduction
        if self.use_svd:
            self.svd = TruncatedSVD(
                n_components=min(self.n_components, self.max_features),
                random_state=42
            )
    
    def fit(self, texts):
        """
        Fit the feature extractor on training data
        
        Parameters:
        -----------
        texts : array-like
            Training text data
            
        Returns:
        --------
        self : Fitted feature extractor
        """
        print(f"Fitting feature extractor on {len(texts)} documents...")
        
        # Fit vectorizer
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Vocabulary size: {len(self.feature_names)}")
        
        # Apply SVD if enabled
        if self.use_svd:
            print(f"Applying SVD dimensionality reduction...")
            X = self.svd.fit_transform(X)
            print(f"Reduced to {X.shape[1]} components")
        
        self.is_fitted = True
        
        return self
    
    def transform(self, texts):
        """
        Transform text data using fitted extractor
        
        Parameters:
        -----------
        texts : array-like
            Text data to transform
            
        Returns:
        --------
        numpy.ndarray : Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transforming")
        
        # Transform text
        X = self.vectorizer.transform(texts)
        
        # Apply SVD if enabled
        if self.use_svd:
            X = self.svd.transform(X)
        
        return X
    
    def fit_transform(self, texts):
        """
        Fit and transform in one step
        
        Parameters:
        -----------
        texts : array-like
            Text data
            
        Returns:
        --------
        numpy.ndarray : Feature matrix
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_importance(self, model, top_n=20):
        """
        Get top N most important features from a trained model
        
        Parameters:
        -----------
        model : sklearn model
            Trained model with feature_importances_ or coef_ attribute
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        pandas.DataFrame : Top features and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted first")
        
        # Get feature importances based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) > 1:
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_)
        else:
            raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
        
        # Get feature names
        if self.use_svd:
            feature_names = [f'SVD_Component_{i}' for i in range(len(importances))]
        else:
            # Ensure we have the right number of feature names
            if len(importances) > len(self.feature_names):
                # For n-grams, we might have more features than names
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            else:
                feature_names = self.feature_names[:len(importances)]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def analyze_features(self, texts, labels=None, top_n=50):
        """
        Analyze extracted features
        
        Parameters:
        -----------
        texts : array-like
            Text data
        labels : array-like, optional
            Labels for feature analysis by class
        top_n : int
            Number of top features to analyze
            
        Returns:
        --------
        dict : Feature analysis results
        """
        if not self.is_fitted:
            X = self.fit_transform(texts)
        else:
            X = self.transform(texts)
        
        analysis = {}
        
        if self.use_svd:
            analysis['method'] = f'SVD-TFIDF ({self.n_components} components)'
            analysis['features'] = [f'Component_{i}' for i in range(X.shape[1])]
        else:
            analysis['method'] = self.method.upper()
            analysis['features'] = self.feature_names.tolist()
        
        analysis['feature_matrix_shape'] = X.shape
        analysis['sparsity'] = 1.0 - (np.count_nonzero(X) / X.size)
        
        # Analyze feature distribution if labels are provided
        if labels is not None and not self.use_svd:
            # Get top features by class
            unique_labels = np.unique(labels)
            class_features = {}
            
            for label in unique_labels:
                class_indices = np.where(labels == label)[0]
                if len(class_indices) > 0:
                    class_X = X[class_indices]
                    # Sum feature values for this class
                    feature_sums = np.array(class_X.sum(axis=0)).flatten()
                    top_indices = np.argsort(feature_sums)[-top_n:][::-1]
                    
                    class_features[f'class_{label}'] = {
                        'top_features': self.feature_names[top_indices].tolist(),
                        'feature_weights': feature_sums[top_indices].tolist()
                    }
            
            analysis['class_features'] = class_features
        
        return analysis
    
    def save(self, filepath):
        """
        Save the feature extractor
        
        Parameters:
        -----------
        filepath : str
            Path to save the extractor
        """
        save_data = {
            'vectorizer': self.vectorizer,
            'svd': self.svd if self.use_svd else None,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'config': {
                'max_features': self.max_features,
                'method': self.method,
                'ngram_range': self.ngram_range,
                'use_svd': self.use_svd,
                'n_components': self.n_components
            }
        }
        
        joblib.dump(save_data, filepath)
        print(f"Feature extractor saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a saved feature extractor
        
        Parameters:
        -----------
        filepath : str
            Path to load the extractor from
        """
        saved_data = joblib.load(filepath)
        
        self.vectorizer = saved_data['vectorizer']
        self.svd = saved_data['svd']
        self.feature_names = saved_data['feature_names']
        self.is_fitted = saved_data['is_fitted']
        
        # Update configuration
        config = saved_data['config']
        self.max_features = config['max_features']
        self.method = config['method']
        self.ngram_range = config['ngram_range']
        self.use_svd = config['use_svd']
        self.n_components = config['n_components']
        
        print(f"Feature extractor loaded from {filepath}")
        return self

# Advanced feature extractor with multiple methods
class AdvancedFeatureExtractor:
    """Advanced feature extractor combining multiple methods"""
    
    def __init__(self, tfidf_params=None, word2vec_params=None, 
                 fasttext_params=None, use_lda=False, n_topics=10):
        """
        Initialize advanced feature extractor
        
        Parameters:
        -----------
        tfidf_params : dict, optional
            Parameters for TF-IDF
        word2vec_params : dict, optional
            Parameters for Word2Vec
        fasttext_params : dict, optional
            Parameters for FastText
        use_lda : bool
            Whether to use LDA topic modeling
        n_topics : int
            Number of topics for LDA
        """
        self.tfidf_params = tfidf_params or {
            'max_features': 3000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95
        }
        
        self.use_lda = use_lda
        self.n_topics = n_topics
        
        # Initialize extractors
        self.tfidf_extractor = FeatureExtractor(**tfidf_params)
        
        if use_lda:
            self.lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                learning_method='online'
            )
        
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def extract_tfidf_features(self, texts, fit=True):
        """Extract TF-IDF features"""
        if fit:
            return self.tfidf_extractor.fit_transform(texts)
        else:
            return self.tfidf_extractor.transform(texts)
    
    def extract_lda_features(self, texts, fit=True):
        """Extract LDA topic features"""
        if not self.use_lda:
            return None
        
        # Transform texts using TF-IDF
        tfidf_features = self.tfidf_extractor.vectorizer.transform(texts)
        
        if fit:
            return self.lda.fit_transform(tfidf_features)
        else:
            return self.lda.transform(tfidf_features)
    
    def extract_all_features(self, texts, fit=True, include_tfidf=True, 
                           include_lda=True, include_meta=True):
        """
        Extract all features
        
        Parameters:
        -----------
        texts : array-like
            Text data
        fit : bool
            Whether to fit the extractors
        include_tfidf : bool
            Whether to include TF-IDF features
        include_lda : bool
            Whether to include LDA features
        include_meta : bool
            Whether to include metadata features
            
        Returns:
        --------
        numpy.ndarray : Combined feature matrix
        """
        feature_blocks = []
        
        # Extract TF-IDF features
        if include_tfidf:
            print("Extracting TF-IDF features...")
            tfidf_features = self.extract_tfidf_features(texts, fit)
            feature_blocks.append(tfidf_features)
            self.feature_names.extend(
                [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
        
        # Extract LDA features
        if include_lda and self.use_lda:
            print("Extracting LDA topic features...")
            lda_features = self.extract_lda_features(texts, fit)
            if lda_features is not None:
                feature_blocks.append(lda_features)
                self.feature_names.extend(
                    [f'topic_{i}' for i in range(lda_features.shape[1])]
                )
        
        # Combine all features
        if feature_blocks:
            # Convert sparse matrices to dense if needed
            dense_blocks = []
            for block in feature_blocks:
                if hasattr(block, 'toarray'):
                    dense_blocks.append(block.toarray())
                else:
                    dense_blocks.append(block)
            
            combined_features = np.hstack(dense_blocks)
            
            # Scale features
            if fit:
                combined_features = self.scaler.fit_transform(combined_features)
            else:
                combined_features = self.scaler.transform(combined_features)
            
            print(f"Extracted {combined_features.shape[1]} total features")
            return combined_features
        else:
            raise ValueError("No feature extraction methods selected")

# Usage example
if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "Machine learning is amazing for text analysis",
        "Natural language processing helps understand human language",
        "Deep learning models require lots of data"
    ]
    
    # Basic feature extractor
    extractor = FeatureExtractor(max_features=100, method='tfidf')
    features = extractor.fit_transform(sample_texts)
    print(f"Feature matrix shape: {features.shape}")
    
    # Advanced feature extractor
    advanced_extractor = AdvancedFeatureExtractor(
        tfidf_params={'max_features': 50},
        use_lda=True,
        n_topics=3
    )
    
    advanced_features = advanced_extractor.extract_all_features(sample_texts)
    print(f"Advanced features shape: {advanced_features.shape}")
