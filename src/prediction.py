import pandas as pd
import numpy as np
import joblib
import re
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NewsPredictor:
    """Main predictor for fake news detection"""
    
    def __init__(self, model_path: str, vectorizer_path: str, 
                 preprocessor_path: Optional[str] = None):
        """
        Initialize predictor with trained components
        
        Parameters:
        -----------
        model_path : str
            Path to trained model
        vectorizer_path : str
            Path to feature vectorizer
        preprocessor_path : str, optional
            Path to text preprocessor
        """
        # Load components
        self.model = self._load_component(model_path, 'model')
        self.vectorizer = self._load_component(vectorizer_path, 'vectorizer')
        
        if preprocessor_path:
            self.preprocessor = self._load_component(preprocessor_path, 'preprocessor')
        else:
            self.preprocessor = None
        
        # Initialize prediction history
        self.prediction_history = []
        
        # Prediction thresholds
        self.thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
        
        # Fake news indicators
        self.fake_indicators = [
            'breaking', 'shocking', 'secret', 'cover-up', 'exposed',
            'miracle', 'instant', 'guaranteed', 'they don\'t want you to know',
            'mainstream media', 'big pharma', 'government hiding'
        ]
        
        # Real news indicators
        self.real_indicators = [
            'according to study', 'research shows', 'published in',
            'peer-reviewed', 'expert said', 'data indicates',
            'official report', 'clinical trial', 'scientists found'
        ]
        
        print(f"Predictor initialized with {type(self.model).__name__}")
    
    def _load_component(self, filepath: str, component_type: str):
        """Load a component from file"""
        try:
            component = joblib.load(filepath)
            print(f"Loaded {component_type} from {filepath}")
            return component
        except Exception as e:
            raise ValueError(f"Error loading {component_type} from {filepath}: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for prediction
        
        Parameters:
        -----------
        text : str
            Raw text input
            
        Returns:
        --------
        str : Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Use custom preprocessor if available
        if self.preprocessor and hasattr(self.preprocessor, 'preprocess_pipeline'):
            return self.preprocessor.preprocess_pipeline(text)
        
        # Default preprocessing
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text: str, use_preprocessor: bool = True):
        """
        Extract features from text
        
        Parameters:
        -----------
        text : str
            Input text
        use_preprocessor : bool
            Whether to use the preprocessor
            
        Returns:
        --------
        array-like : Feature vector
        """
        if use_preprocessor:
            processed_text = self.preprocess_text(text)
        else:
            processed_text = text
        
        # Transform text to features
        features = self.vectorizer.transform([processed_text])
        
        return features
    
    def analyze_text_indicators(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for fake/real news indicators
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        dict : Indicator analysis
        """
        text_lower = text.lower()
        
        # Count indicators
        fake_count = sum(1 for indicator in self.fake_indicators 
                        if indicator in text_lower)
        real_count = sum(1 for indicator in self.real_indicators 
                        if indicator in text_lower)
        
        # Calculate ratios
        total_indicators = fake_count + real_count
        if total_indicators > 0:
            fake_ratio = fake_count / total_indicators
            real_ratio = real_count / total_indicators
        else:
            fake_ratio = real_ratio = 0.5
        
        # Text statistics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        # Check for sensational language
        sensational_words = ['!', 'BREAKING', 'SHOCKING', 'URGENT']
        sensational_score = sum(text.count(word) for word in sensational_words)
        
        return {
            'fake_indicators': fake_count,
            'real_indicators': real_count,
            'fake_ratio': fake_ratio,
            'real_ratio': real_ratio,
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'sensational_score': sensational_score,
            'indicator_balance': 'fake' if fake_ratio > 0.7 else 'real' if real_ratio > 0.7 else 'neutral'
        }
    
    def predict_single(self, text: str, return_analysis: bool = True, 
                      threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Make prediction for a single text
        
        Parameters:
        -----------
        text : str
            Input text
        return_analysis : bool
            Whether to return detailed analysis
        threshold : float, optional
            Custom classification threshold (0-1)
            
        Returns:
        --------
        dict : Prediction results
        """
        if not text or len(text.strip()) < 10:
            return {
                'error': 'Text is too short',
                'text': text[:100] + '...' if len(text) > 100 else text
            }
        
        # Extract features
        features = self.extract_features(text)
        
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            # Get probability predictions
            probabilities = self.model.predict_proba(features)[0]
            
            # Determine class based on threshold
            if threshold is not None:
                prediction = 1 if probabilities[1] >= threshold else 0
                confidence = probabilities[1] if prediction == 1 else probabilities[0]
            else:
                prediction = self.model.predict(features)[0]
                confidence = max(probabilities)
            
            prob_real, prob_fake = probabilities[0], probabilities[1]
            
        else:
            # Model doesn't support probabilities
            prediction = self.model.predict(features)[0]
            confidence = 1.0  # Assume maximum confidence
            prob_real = 1.0 if prediction == 0 else 0.0
            prob_fake = 1.0 if prediction == 1 else 0.0
        
        # Get prediction label
        label = 'fake' if prediction == 1 else 'real'
        
        # Determine confidence level
        if confidence >= self.thresholds['high_confidence']:
            confidence_level = 'high'
        elif confidence >= self.thresholds['medium_confidence']:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        # Create result dictionary
        result = {
            'text': text[:500] + '...' if len(text) > 500 else text,  # Truncate for display
            'prediction': label,
            'confidence': float(confidence),
            'confidence_level': confidence_level,
            'probability': {
                'real': float(prob_real),
                'fake': float(prob_fake)
            },
            'timestamp': datetime.now().isoformat(),
            'model': type(self.model).__name__
        }
        
        # Add detailed analysis if requested
        if return_analysis:
            indicators = self.analyze_text_indicators(text)
            result['analysis'] = indicators
            
            # Combine model prediction with indicator analysis
            indicator_prediction = 'fake' if indicators['fake_ratio'] > 0.7 else 'real' if indicators['real_ratio'] > 0.7 else 'uncertain'
            
            result['analysis']['indicator_prediction'] = indicator_prediction
            result['analysis']['agreement'] = 'yes' if label == indicator_prediction else 'no'
        
        # Add to history
        self.prediction_history.append(result)
        
        return result
    
    def predict_batch(self, texts: List[str], return_analysis: bool = False,
                     threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple texts
        
        Parameters:
        -----------
        texts : list
            List of text strings
        return_analysis : bool
            Whether to return detailed analysis
        threshold : float, optional
            Custom classification threshold
            
        Returns:
        --------
        list : List of prediction results
        """
        results = []
        
        print(f"Processing {len(texts)} texts...")
        
        for i, text in enumerate(texts, 1):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(texts)} texts...")
            
            result = self.predict_single(text, return_analysis, threshold)
            results.append(result)
        
        print(f"Completed processing {len(texts)} texts")
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'text',
                         return_analysis: bool = False, 
                         threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Make predictions for a DataFrame
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        text_column : str
            Name of the text column
        return_analysis : bool
            Whether to return detailed analysis
        threshold : float, optional
            Custom classification threshold
            
        Returns:
        --------
        pandas.DataFrame : DataFrame with predictions
        """
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")
        
        print(f"Processing DataFrame with {len(df)} rows...")
        
        predictions = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            
            # Make prediction
            result = self.predict_single(text, return_analysis, threshold)
            
            # Add original data
            result.update({col: row[col] for col in df.columns if col != text_column})
            
            predictions.append(result)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(predictions)
        
        # Reorder columns
        cols = ['prediction', 'confidence', 'confidence_level'] + [c for c in result_df.columns 
                                                                  if c not in ['prediction', 'confidence', 'confidence_level']]
        result_df = result_df[cols]
        
        # Add statistics
        fake_count = (result_df['prediction'] == 'fake').sum()
        real_count = (result_df['prediction'] == 'real').sum()
        
        print(f"\nPrediction Summary:")
        print(f"  Fake news: {fake_count} ({fake_count/len(df)*100:.1f}%)")
        print(f"  Real news: {real_count} ({real_count/len(df)*100:.1f}%)")
        
        if 'confidence' in result_df.columns:
            avg_confidence = result_df['confidence'].mean()
            print(f"  Average confidence: {avg_confidence:.3f}")
        
        return result_df
    
    def explain_prediction(self, text: str, top_features: int = 10) -> Dict[str, Any]:
        """
        Explain the prediction by showing important features
        
        Parameters:
        -----------
        text : str
            Input text
        top_features : int
            Number of top features to show
            
        Returns:
        --------
        dict : Explanation with important features
        """
        # Get prediction
        prediction = self.predict_single(text, return_analysis=True)
        
        # Check if model supports feature importance
        if not hasattr(self.model, 'coef_') and not hasattr(self.model, 'feature_importances_'):
            return {
                'error': 'Model does not support feature importance explanation',
                'prediction': prediction
            }
        
        # Get feature names
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            feature_names = self.vectorizer.get_feature_names_out()
        elif hasattr(self.vectorizer, 'get_feature_names'):
            feature_names = self.vectorizer.get_feature_names()
        else:
            return {
                'error': 'Vectorizer does not provide feature names',
                'prediction': prediction
            }
        
        # Get feature importance
        if hasattr(self.model, 'coef_'):
            # Linear model
            importance = self.model.coef_[0]
        else:
            # Tree-based model
            importance = self.model.feature_importances_
        
        # Transform text to get feature values
        features = self.extract_features(text)
        
        if hasattr(features, 'toarray'):
            feature_values = features.toarray()[0]
        else:
            feature_values = features[0]
        
        # Create feature importance DataFrame
        feature_data = []
        for i, (name, value, imp) in enumerate(zip(feature_names, feature_values, importance)):
            if value != 0:  # Only include features present in text
                contribution = value * imp
                feature_data.append({
                    'feature': name,
                    'value': float(value),
                    'importance': float(imp),
                    'contribution': float(contribution),
                    'direction': 'supports_fake' if contribution > 0 else 'supports_real'
                })
        
        # Sort by absolute contribution
        feature_df = pd.DataFrame(feature_data)
        feature_df['abs_contribution'] = feature_df['contribution'].abs()
        feature_df = feature_df.sort_values('abs_contribution', ascending=False).head(top_features)
        
        # Get top features for each class
        fake_features = feature_df[feature_df['direction'] == 'supports_fake'].head(5)
        real_features = feature_df[feature_df['direction'] == 'supports_real'].head(5)
        
        explanation = {
            'prediction': prediction,
            'top_features': feature_df.to_dict('records'),
            'summary': {
                'features_supporting_fake': len(fake_features),
                'features_supporting_real': len(real_features),
                'total_active_features': len(feature_df),
                'dominant_direction': 'fake' if len(fake_features) > len(real_features) else 'real'
            },
            'key_indicators': {
                'fake': fake_features['feature'].tolist() if not fake_features.empty else [],
                'real': real_features['feature'].tolist() if not real_features.empty else []
            }
        }
        
        return explanation
    
    def get_prediction_history(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get prediction history
        
        Parameters:
        -----------
        limit : int, optional
            Maximum number of history entries to return
            
        Returns:
        --------
        pandas.DataFrame : Prediction history
        """
        if not self.prediction_history:
            return pd.DataFrame()
        
        history_df = pd.DataFrame(self.prediction_history)
        
        if limit:
            history_df = history_df.tail(limit)
        
        return history_df
    
    def save_prediction_history(self, filepath: str):
        """Save prediction history to file"""
        if not self.prediction_history:
            print("No prediction history to save")
            return
        
        history_df = self.get_prediction_history()
        history_df.to_csv(filepath, index=False)
        print(f"Prediction history saved to {filepath}")
    
    def clear_prediction_history(self):
        """Clear prediction history"""
        self.prediction_history = []
        print("Prediction history cleared")
    
    def evaluate_on_dataset(self, df: pd.DataFrame, text_column: str = 'text',
                          label_column: str = 'label') -> Dict[str, Any]:
        """
        Evaluate predictor on a labeled dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Labeled dataset
        text_column : str
            Name of text column
        label_column : str
            Name of label column
            
        Returns:
        --------
        dict : Evaluation results
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in DataFrame")
        
        print(f"Evaluating on {len(df)} samples...")
        
        # Make predictions
        predictions = self.predict_dataframe(df, text_column, return_analysis=False)
        
        # Convert predictions to numeric labels
        pred_labels = predictions['prediction'].map({'fake': 1, 'real': 0}).values
        true_labels = df[label_column].values
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, zero_division=0),
            'recall': recall_score(true_labels, pred_labels, zero_division=0),
            'f1_score': f1_score(true_labels, pred_labels, zero_division=0),
            'total_samples': len(df),
            'correct_predictions': (pred_labels == true_labels).sum(),
            'incorrect_predictions': (pred_labels != true_labels).sum()
        }
        
        # Add confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, pred_labels)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negative'] = int(cm[0, 0])
        metrics['false_positive'] = int(cm[0, 1])
        metrics['false_negative'] = int(cm[1, 0])
        metrics['true_positive'] = int(cm[1, 1])
        
        print(f"\nEvaluation Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        print(f"  Correct:   {metrics['correct_predictions']}/{metrics['total_samples']}")
        
        return metrics

# Advanced predictor with ensemble support
class EnsemblePredictor(NewsPredictor):
    """Predictor that supports ensemble models"""
    
    def __init__(self, model_paths: List[str], vectorizer_path: str,
                 preprocessor_path: Optional[str] = None, voting: str = 'soft'):
        """
        Initialize ensemble predictor
        
        Parameters:
        -----------
        model_paths : list
            List of paths to trained models
        vectorizer_path : str
            Path to feature vectorizer
        preprocessor_path : str, optional
            Path to text preprocessor
        voting : str
            Voting method ('hard' or 'soft')
        """
        # Load models
        self.models = [self._load_component(path, f'model_{i}') 
                      for i, path in enumerate(model_paths)]
        self.vectorizer = self._load_component(vectorizer_path, 'vectorizer')
        
        if preprocessor_path:
            self.preprocessor = self._load_component(preprocessor_path, 'preprocessor')
        else:
            self.preprocessor = None
        
        self.voting = voting
        self.model_names = [type(model).__name__ for model in self.models]
        
        print(f"Ensemble predictor initialized with {len(self.models)} models")
        print(f"Models: {', '.join(self.model_names)}")
        print(f"Voting method: {voting}")
    
    def predict_single(self, text: str, return_analysis: bool = True,
                      threshold: Optional[float] = None) -> Dict[str, Any]:
        """Make prediction using ensemble"""
        if self.voting == 'hard':
            return self._hard_voting(text, return_analysis, threshold)
        else:
            return self._soft_voting(text, return_analysis, threshold)
    
    def _hard_voting(self, text: str, return_analysis: bool,
                    threshold: Optional[float]) -> Dict[str, Any]:
        """Hard voting prediction"""
        # Get predictions from all models
        predictions = []
        probabilities = []
        
        for model in self.models:
            features = self.extract_features(text)
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features)[0]
                pred = 1 if prob[1] >= (threshold or 0.5) else 0
                probabilities.append(prob)
            else:
                pred = model.predict(features)[0]
                prob = [1.0, 0.0] if pred == 0 else [0.0, 1.0]
                probabilities.append(prob)
            
            predictions.append(pred)
        
        # Majority vote
        final_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0
        
        # Average probabilities
        avg_prob = np.mean(probabilities, axis=0)
        
        return self._format_result(text, final_prediction, avg_prob, return_analysis)
    
    def _soft_voting(self, text: str, return_analysis: bool,
                    threshold: Optional[float]) -> Dict[str, Any]:
        """Soft voting prediction"""
        # Average probabilities from all models
        all_probabilities = []
        
        for model in self.models:
            features = self.extract_features(text)
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features)[0]
            else:
                pred = model.predict(features)[0]
                prob = [1.0, 0.0] if pred == 0 else [0.0, 1.0]
            
            all_probabilities.append(prob)
        
        # Average probabilities
        avg_prob = np.mean(all_probabilities, axis=0)
        
        # Determine final prediction
        if threshold is not None:
            final_prediction = 1 if avg_prob[1] >= threshold else 0
        else:
            final_prediction = 1 if avg_prob[1] > avg_prob[0] else 0
        
        return self._format_result(text, final_prediction, avg_prob, return_analysis)
    
    def _format_result(self, text: str, prediction: int, probabilities: np.ndarray,
                      return_analysis: bool) -> Dict[str, Any]:
        """Format prediction result"""
        label = 'fake' if prediction == 1 else 'real'
        confidence = max(probabilities)
        
        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = 'high'
        elif confidence >= 0.6:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        result = {
            'text': text[:500] + '...' if len(text) > 500 else text,
            'prediction': label,
            'confidence': float(confidence),
            'confidence_level': confidence_level,
            'probability': {
                'real': float(probabilities[0]),
                'fake': float(probabilities[1])
            },
            'timestamp': datetime.now().isoformat(),
            'model': f'Ensemble ({len(self.models)} models)',
            'voting_method': self.voting,
            'model_names': self.model_names
        }
        
        if return_analysis:
            indicators = self.analyze_text_indicators(text)
            result['analysis'] = indicators
        
        # Add to history
        self.prediction_history.append(result)
        
        return result

# Real-time predictor for web applications
class RealTimePredictor:
    """Real-time predictor optimized for web applications"""
    
    def __init__(self, model_path: str, vectorizer_path: str,
                 cache_size: int = 1000):
        """
        Initialize real-time predictor
        
        Parameters:
        -----------
        model_path : str
            Path to trained model
        vectorizer_path : str
            Path to feature vectorizer
        cache_size : int
            Size of prediction cache
        """
        self.predictor = NewsPredictor(model_path, vectorizer_path)
        self.cache = {}
        self.cache_size = cache_size
        self.request_count = 0
        
    def predict(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Make prediction with caching
        
        Parameters:
        -----------
        text : str
            Input text
        use_cache : bool
            Whether to use cache
            
        Returns:
        --------
        dict : Prediction result
        """
        self.request_count += 1
        
        # Create cache key (hash of text)
        import hashlib
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        if use_cache and cache_key in self.cache:
            result = self.cache[cache_key].copy()
            result['cached'] = True
            return result
        
        # Make new prediction
        result = self.predictor.predict_single(text, return_analysis=False)
        result['cached'] = False
        result['request_id'] = self.request_count
        
        # Update cache
        self.cache[cache_key] = result
        
        # Limit cache size
        if len(self.cache) > self.cache_size:
            # Remove oldest entry (simplified)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get predictor statistics"""
        return {
            'total_requests': self.request_count,
            'cache_size': len(self.cache),
            'cache_hits': sum(1 for r in self.predictor.prediction_history 
                            if r.get('cached', False)),
            'cache_misses': self.request_count - sum(1 for r in self.predictor.prediction_history 
                                                   if r.get('cached', False)),
            'average_response_time': 0.01  # Placeholder
        }

# Usage example
if __name__ == "__main__":
    # Example usage
    print("Testing NewsPredictor...")
    
    # Sample texts
    sample_texts = [
        "BREAKING: Miracle cure discovered for all diseases! Doctors hate this!",
        "According to a new study published in Nature, regular exercise improves health.",
        "The government is hiding the truth about aliens visiting Earth!"
    ]
    
    # Note: In real usage, you need to load actual trained models
    # For demonstration, we'll create a mock predictor
    
    class MockModel:
        def predict(self, X):
            return np.random.randint(0, 2, size=X.shape[0])
        
        def predict_proba(self, X):
            prob = np.random.rand(X.shape[0], 2)
            return prob / prob.sum(axis=1, keepdims=1)
    
    class MockVectorizer:
        def transform(self, texts):
            return np.random.rand(len(texts), 10)
    
    # Create mock predictor for demonstration
    print("\nMock prediction example:")
    for text in sample_texts:
        print(f"\nText: {text[:50]}...")
        print("Prediction would be made here with real model")
    
    print("\nTo use real predictor:")
    print("1. Train and save your model")
    print("2. Initialize predictor: predictor = NewsPredictor('model.pkl', 'vectorizer.pkl')")
    print("3. Make predictions: result = predictor.predict_single('Your news text here')")
