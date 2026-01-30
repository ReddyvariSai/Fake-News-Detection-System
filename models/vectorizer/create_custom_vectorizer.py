import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re
from datetime import datetime

class FakeNewsVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom vectorizer specifically designed for fake news detection
    Combines multiple feature extraction techniques
    """
    
    def __init__(self, max_features=5000):
        self.max_features = max_features
        
        # Word-level TF-IDF
        self.word_vectorizer = TfidfVectorizer(
            max_features=max_features // 2,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95,
            analyzer='word',
            token_pattern=r'(?u)\b\w\w+\b',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        # Character n-grams (captures patterns like "!!", "BREAKING")
        self.char_vectorizer = TfidfVectorizer(
            max_features=max_features // 4,
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=2
        )
        
        # Binary features (presence/absence)
        self.binary_vectorizer = CountVectorizer(
            max_features=max_features // 4,
            binary=True,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Metadata
        self.metadata = {
            'created': datetime.now().isoformat(),
            'type': 'FakeNewsVectorizer',
            'version': '2.0.0',
            'description': 'Custom vectorizer for fake news detection'
        }
        
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Fit all vectorizers"""
        print("Fitting FakeNewsVectorizer...")
        
        self.word_vectorizer.fit(X)
        self.char_vectorizer.fit(X)
        self.binary_vectorizer.fit(X)
        
        # Extract fake news specific vocabulary
        if y is not None:
            self._extract_fake_news_vocabulary(X, y)
        
        self.is_fitted = True
        
        # Update metadata
        self.metadata.update({
            'vocabulary_sizes': {
                'word_features': len(self.word_vectorizer.vocabulary_),
                'char_features': len(self.char_vectorizer.vocabulary_),
                'binary_features': len(self.binary_vectorizer.vocabulary_),
                'total': (len(self.word_vectorizer.vocabulary_) +
                         len(self.char_vectorizer.vocabulary_) +
                         len(self.binary_vectorizer.vocabulary_))
            },
            'parameters': {
                'max_features': self.max_features,
                'word_params': self.word_vectorizer.get_params(),
                'char_params': self.char_vectorizer.get_params(),
                'binary_params': self.binary_vectorizer.get_params()
            }
        })
        
        return self
    
    def _extract_fake_news_vocabulary(self, X, y):
        """Extract vocabulary specific to fake news"""
        from collections import Counter
        
        # Separate real and fake texts
        real_texts = [text for text, label in zip(X, y) if label == 0]
        fake_texts = [text for text, label in zip(X, y) if label == 1]
        
        # Extract words
        real_words = self._extract_words(real_texts)
        fake_words = self._extract_words(fake_texts)
        
        # Find words more common in fake news
        fake_word_counts = Counter(fake_words)
        real_word_counts = Counter(real_words)
        
        fake_specific = {}
        for word, fake_count in fake_word_counts.items():
            real_count = real_word_counts.get(word, 0)
            fake_ratio = fake_count / max(real_count, 1)
            if fake_ratio > 2.0 and fake_count > 2:  # 2x more common in fake news
                fake_specific[word] = {
                    'fake_count': fake_count,
                    'real_count': real_count,
                    'ratio': fake_ratio
                }
        
        self.fake_news_vocabulary = fake_specific
    
    def _extract_words(self, texts):
        """Extract words from texts"""
        words = []
        for text in texts:
            text_lower = text.lower()
            # Simple word extraction
            tokens = re.findall(r'\b\w+\b', text_lower)
            words.extend(tokens)
        return words
    
    def transform(self, X):
        """Transform texts using all vectorizers"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        # Get transformations from each vectorizer
        word_features = self.word_vectorizer.transform(X)
        char_features = self.char_vectorizer.transform(X)
        binary_features = self.binary_vectorizer.transform(X)
        
        # Combine features
        from scipy import sparse
        combined = sparse.hstack([word_features, char_features, binary_features])
        
        return combined
    
    def fit_transform(self, X, y=None):
        """Fit and transform"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self):
        """Get combined feature names"""
        word_names = [f"word_{name}" for name in 
                     self.word_vectorizer.get_feature_names_out()]
        char_names = [f"char_{name}" for name in 
                     self.char_vectorizer.get_feature_names_out()]
        binary_names = [f"bin_{name}" for name in 
                       self.binary_vectorizer.get_feature_names_out()]
        
        return word_names + char_names + binary_names
    
    def analyze_text(self, text):
        """Analyze a text and return feature breakdown"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        
        # Transform the text
        features = self.transform([text])
        
        # Get feature names and values
        feature_names = self.get_feature_names()
        
        # Find non-zero features
        row = features[0]
        indices = row.nonzero()[1]
        values = row.data
        
        # Organize by feature type
        breakdown = {
            'word_features': [],
            'char_features': [],
            'binary_features': [],
            'total_features': len(indices),
            'fake_news_indicators': []
        }
        
        for idx, val in zip(indices, values):
            name = feature_names[idx]
            value = float(val)
            
            # Categorize
            if name.startswith('word_'):
                breakdown['word_features'].append({
                    'feature': name[5:],  # Remove 'word_' prefix
                    'value': value
                })
            elif name.startswith('char_'):
                breakdown['char_features'].append({
                    'feature': name[5:],  # Remove 'char_' prefix
                    'value': value
                })
            elif name.startswith('bin_'):
                breakdown['binary_features'].append({
                    'feature': name[4:],  # Remove 'bin_' prefix
                    'value': value
                })
            
            # Check if this is a fake news indicator
            if hasattr(self, 'fake_news_vocabulary'):
                for fake_word in self.fake_news_vocabulary:
                    if fake_word in name.lower():
                        breakdown['fake_news_indicators'].append({
                            'word': fake_word,
                            'feature': name,
                            'value': value,
                            'ratio': self.fake_news_vocabulary[fake_word]['ratio']
                        })
        
        return breakdown

def create_and_save_custom_vectorizer():
    """Create and save the custom vectorizer"""
    
    print("Creating custom FakeNewsVectorizer...")
    
    # Create vectorizer
    vectorizer = FakeNewsVectorizer(max_features=5000)
    
    # Create sample training data with labels
    sample_texts = [
        # Real news (label 0)
        "scientists publish research on climate change in journal",
        "government releases official economic statistics report",
        "university study confirms benefits of regular exercise",
        "medical researchers announce breakthrough in treatment",
        "financial markets show stability according to analysts",
        
        # Fake news (label 1)
        "BREAKING: Secret cure for all diseases discovered!",
        "SHOCKING government cover-up EXPOSED by insider!",
        "Miracle weight loss trick doctors don't want you to know!",
        "URGENT: They're putting chemicals in our food supply!",
        "AMAZING discovery BIG PHARMA is trying to suppress!",
        
        # More examples
        "new policy aims to improve public healthcare services",
        "technology company innovates in renewable energy sector",
        "CONSPIRACY: The truth about vaccines they're hiding!",
        "research indicates positive trends in economic recovery",
        "ACT NOW: This information will be DELETED soon!"
    ]
    
    sample_labels = [0, 0, 0, 0, 0,  # Real news
                    1, 1, 1, 1, 1,  # Fake news
                    0, 0, 1, 0, 1]  # Mixed
    
    # Fit the vectorizer
    vectorizer.fit(sample_texts, sample_labels)
    
    # Save the vectorizer
    import os
    os.makedirs('models', exist_ok=True)
    
    vectorizer_path = 'models/fake_news_custom_vectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_path, compress=3)
    
    print(f"\n✅ Custom vectorizer saved: {vectorizer_path}")
    print(f"📊 Metadata:")
    print(f"   Type: {vectorizer.metadata['type']}")
    print(f"   Version: {vectorizer.metadata['version']}")
    print(f"   Total features: {vectorizer.metadata['vocabulary_sizes']['total']}")
    
    if hasattr(vectorizer, 'fake_news_vocabulary'):
        print(f"   Fake news specific words: {len(vectorizer.fake_news_vocabulary)}")
        print(f"   Sample fake news words: {list(vectorizer.fake_news_vocabulary.keys())[:5]}")
    
    # Test the vectorizer
    print("\n🧪 Testing vectorizer...")
    
    test_text = "BREAKING NEWS: Secret government documents LEAKED online!"
    analysis = vectorizer.analyze_text(test_text)
    
    print(f"Test text: '{test_text}'")
    print(f"Total features extracted: {analysis['total_features']}")
    print(f"Word features: {len(analysis['word_features'])}")
    print(f"Character features: {len(analysis['char_features'])}")
    print(f"Binary features: {len(analysis['binary_features'])}")
    
    if analysis['fake_news_indicators']:
        print(f"\n🚨 Fake news indicators found:")
        for indicator in analysis['fake_news_indicators'][:3]:
            print(f"   - '{indicator['word']}' (ratio: {indicator['ratio']:.1f}x)")
    
    return vectorizer_path, vectorizer

if __name__ == "__main__":
    create_and_save_custom_vectorizer()
