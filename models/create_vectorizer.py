import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

def create_vectorizer_pkl():
    """Create a complete vectorizer.pkl file"""
    
    print("=" * 60)
    print("CREATING VECTORIZER.PKL FILE")
    print("=" * 60)
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # 1. Create TF-IDF Vectorizer with optimal parameters for fake news detection
    print("\n🔧 Creating TF-IDF Vectorizer...")
    
    vectorizer = TfidfVectorizer(
        # Feature extraction parameters
        max_features=5000,           # Limit vocabulary size
        min_df=2,                    # Ignore terms that appear in less than 2 documents
        max_df=0.95,                 # Ignore terms that appear in more than 95% of documents
        stop_words='english',        # Remove common English stop words
        
        # N-gram parameters (capture phrases and word combinations)
        ngram_range=(1, 3),          # Use unigrams, bigrams, and trigrams
        
        # Text processing parameters
        lowercase=True,              # Convert all characters to lowercase
        strip_accents='unicode',     # Remove accents and special characters
        analyzer='word',             # Tokenize by word
        
        # TF-IDF parameters
        use_idf=True,                # Use Inverse Document Frequency
        smooth_idf=True,             # Smooth IDF weights
        sublinear_tf=True,           # Apply sublinear TF scaling (1 + log(tf))
        
        # Special parameters for fake news detection
        token_pattern=r'(?u)\b\w\w+\b',  # Token pattern (words with 2+ chars)
        norm='l2',                        # Normalize vectors (Euclidean norm)
    )
    
    # 2. Create sample training data (typical of fake news datasets)
    print("\n📊 Creating sample training data...")
    
    sample_texts = [
        # Real news examples (more factual, less sensational)
        "scientists discover new evidence of climate change in latest study",
        "government releases economic data showing growth in employment numbers",
        "research published in peer reviewed journal confirms vaccine effectiveness",
        "central bank announces interest rate decision based on inflation data",
        "official statistics indicate decrease in poverty rates over past decade",
        "medical trial results show promising treatment for common disease",
        "university study finds correlation between exercise and mental health",
        "financial regulators implement new rules to protect consumer interests",
        "environmental agency reports improvement in air quality measurements",
        "international organization releases guidelines for sustainable development",
        
        # Fake news examples (more sensational, emotional language)
        "breaking shocking discovery cures all diseases overnight doctors furious",
        "secret government cover up exposed anonymous insider reveals truth",
        "miracle supplement banned by big pharma because it works too well",
        "they dont want you to know this simple trick to lose weight fast",
        "elite conspiracy revealed mainstream media hiding important information",
        "one weird herb destroys cancer cells in hours clinical proof suppressed",
        "urgent warning they are putting dangerous chemicals in our food supply",
        "hidden truth about vaccines exposed whistleblower speaks out now",
        "amazing breakthrough in energy technology being suppressed by corporations",
        "you wont believe what celebrity said about controversial political issue",
        
        # More balanced examples
        "new policy implemented to address concerns raised by community members",
        "technology company announces innovation in renewable energy sector",
        "educational institution receives funding for research initiatives",
        "health officials provide updated guidance based on latest evidence",
        "economic indicators suggest stable growth despite global challenges",
        
        # Additional fake news markers
        "act now before its too late this information will be deleted soon",
        "share this with everyone you know before they take it down",
        "doctors hate this one simple trick for instant health improvement",
        "the establishment doesnt want you to see these leaked documents",
        "mainstream media refuses to report this important breaking news"
    ]
    
    # Fit the vectorizer on sample data
    print("🏋️ Training vectorizer on sample texts...")
    vectorizer.fit(sample_texts)
    
    # Get vocabulary information
    vocabulary = vectorizer.get_feature_names_out()
    print(f"✅ Vectorizer trained successfully!")
    print(f"   Vocabulary size: {len(vocabulary)} words/phrases")
    print(f"   Max features: {vectorizer.max_features}")
    print(f"   N-gram range: {vectorizer.ngram_range}")
    
    # Show some sample features
    print(f"   Sample features (first 10): {vocabulary[:10]}")
    
    # 3. Add metadata to the vectorizer object
    print("\n📝 Adding metadata to vectorizer...")
    
    vectorizer.metadata = {
        'vectorizer_type': 'TF-IDF Vectorizer',
        'created_date': datetime.now().isoformat(),
        'version': '1.0.0',
        'description': 'Text vectorizer for fake news detection',
        'parameters': {
            'max_features': vectorizer.max_features,
            'min_df': vectorizer.min_df,
            'max_df': vectorizer.max_df,
            'ngram_range': vectorizer.ngram_range,
            'stop_words': 'english',
            'use_idf': vectorizer.use_idf,
            'smooth_idf': vectorizer.smooth_idf,
            'sublinear_tf': vectorizer.sublinear_tf,
            'norm': vectorizer.norm
        },
        'vocabulary_stats': {
            'total_features': len(vocabulary),
            'unigrams': len([f for f in vocabulary if ' ' not in f]),
            'bigrams': len([f for f in vocabulary if len(f.split()) == 2]),
            'trigrams': len([f for f in vocabulary if len(f.split()) == 3]),
            'sample_features': vocabulary[:20].tolist()
        },
        'training_info': {
            'sample_count': len(sample_texts),
            'real_news_samples': 15,
            'fake_news_samples': 15,
            'balanced_samples': 5
        }
    }
    
    # 4. Save the vectorizer
    print("\n💾 Saving vectorizer to models/vectorizer.pkl...")
    
    vectorizer_path = 'models/vectorizer.pkl'
    
    # Save with joblib (better for scikit-learn objects)
    joblib.dump(vectorizer, vectorizer_path, compress=3)
    
    # Verify the file was created
    file_size = os.path.getsize(vectorizer_path)
    print(f"✅ Vectorizer saved successfully!")
    print(f"   File: {vectorizer_path}")
    print(f"   Size: {file_size:,} bytes")
    
    # 5. Test the vectorizer
    print("\n🧪 Testing vectorizer functionality...")
    
    # Load it back
    loaded_vectorizer = joblib.load(vectorizer_path)
    print(f"✅ Vectorizer loaded back successfully")
    
    # Test transformation
    test_texts = [
        "breaking news amazing discovery cures disease",
        "scientific study confirms correlation between variables"
    ]
    
    transformed = loaded_vectorizer.transform(test_texts)
    print(f"✅ Transformation successful")
    print(f"   Shape of transformed data: {transformed.shape}")
    print(f"   Number of non-zero features: {transformed.nnz}")
    
    # Show some feature names from test texts
    print(f"\n🔍 Feature extraction example:")
    for i, text in enumerate(test_texts):
        transformed_vec = loaded_vectorizer.transform([text])
        non_zero_indices = transformed_vec.nonzero()[1]
        feature_names = loaded_vectorizer.get_feature_names_out()[non_zero_indices]
        
        print(f"\n   Text {i+1}: '{text[:50]}...'")
        print(f"   Extracted features ({len(feature_names)}): {feature_names[:5]}...")
    
    return vectorizer_path, loaded_vectorizer

def create_advanced_vectorizer():
    """Create a more advanced vectorizer with additional features"""
    
    print("\n" + "=" * 60)
    print("CREATING ADVANCED VECTORIZER WITH MULTIPLE FEATURES")
    print("=" * 60)
    
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Create an ensemble of vectorizers
    class AdvancedVectorizer:
        """Advanced vectorizer combining multiple feature extraction methods"""
        
        def __init__(self):
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            
            self.char_vectorizer = TfidfVectorizer(
                max_features=1000,
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=2
            )
            
            self.binary_vectorizer = CountVectorizer(
                max_features=1000,
                binary=True,
                stop_words='english'
            )
            
            self.is_fitted = False
            self.metadata = {}
        
        def fit(self, texts):
            """Fit all vectorizers"""
            print("Training advanced vectorizer ensemble...")
            
            self.tfidf_vectorizer.fit(texts)
            self.char_vectorizer.fit(texts)
            self.binary_vectorizer.fit(texts)
            
            self.is_fitted = True
            
            # Create metadata
            self.metadata = {
                'type': 'AdvancedVectorizer',
                'components': {
                    'tfidf_word': self.tfidf_vectorizer.get_params(),
                    'tfidf_char': self.char_vectorizer.get_params(),
                    'binary': self.binary_vectorizer.get_params()
                },
                'total_features': (
                    self.tfidf_vectorizer.max_features +
                    self.char_vectorizer.max_features +
                    self.binary_vectorizer.max_features
                ),
                'created': datetime.now().isoformat()
            }
            
            return self
        
        def transform(self, texts):
            """Transform texts using all vectorizers"""
            if not self.is_fitted:
                raise ValueError("Vectorizer must be fitted before transform")
            
            # Get transformations from each vectorizer
            tfidf_features = self.tfidf_vectorizer.transform(texts)
            char_features = self.char_vectorizer.transform(texts)
            binary_features = self.binary_vectorizer.transform(texts)
            
            # Combine features (horizontal stacking)
            from scipy import sparse
            combined = sparse.hstack([tfidf_features, char_features, binary_features])
            
            return combined
        
        def fit_transform(self, texts):
            """Fit and transform"""
            return self.fit(texts).transform(texts)
        
        def get_feature_names(self):
            """Get combined feature names"""
            tfidf_names = [f"word_{name}" for name in 
                          self.tfidf_vectorizer.get_feature_names_out()]
            char_names = [f"char_{name}" for name in 
                         self.char_vectorizer.get_feature_names_out()]
            binary_names = [f"bin_{name}" for name in 
                           self.binary_vectorizer.get_feature_names_out()]
            
            return tfidf_names + char_names + binary_names
    
    # Create and save advanced vectorizer
    advanced_vectorizer = AdvancedVectorizer()
    
    # Sample training data
    sample_texts = [
        "real news article about scientific research",
        "fake news sensational headline breaking now",
        "government report economic data analysis",
        "conspiracy theory secret information leaked"
    ]
    
    # Fit the vectorizer
    advanced_vectorizer.fit(sample_texts)
    
    # Save it
    advanced_path = 'models/advanced_vectorizer.pkl'
    joblib.dump(advanced_vectorizer, advanced_path, compress=3)
    
    print(f"✅ Advanced vectorizer saved: {advanced_path}")
    print(f"   Total features: {advanced_vectorizer.metadata['total_features']}")
    print(f"   Components: {list(advanced_vectorizer.metadata['components'].keys())}")
    
    return advanced_path

def create_supporting_files():
    """Create additional supporting files for the vectorizer"""
    
    print("\n" + "=" * 60)
    print("CREATING SUPPORTING FILES")
    print("=" * 60)
    
    # 1. Create feature names file
    vectorizer = joblib.load('models/vectorizer.pkl')
    feature_names = vectorizer.get_feature_names_out()
    
    feature_names_path = 'models/feature_names.txt'
    with open(feature_names_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    print(f"✅ Feature names saved: {feature_names_path}")
    print(f"   Total features: {len(feature_names)}")
    
    # 2. Create vocabulary statistics
    import json
    
    vocab_stats = {
        'total_features': len(feature_names),
        'feature_categories': {
            'unigrams': len([f for f in feature_names if ' ' not in f]),
            'bigrams': len([f for f in feature_names if len(f.split()) == 2]),
            'trigrams': len([f for f in feature_names if len(f.split()) == 3]),
        },
        'most_common_patterns': {
            'breaking': sum(1 for f in feature_names if 'breaking' in f),
            'news': sum(1 for f in feature_names if 'news' in f),
            'secret': sum(1 for f in feature_names if 'secret' in f),
            'study': sum(1 for f in feature_names if 'study' in f),
            'government': sum(1 for f in feature_names if 'government' in f),
        },
        'sample_features': feature_names[:100].tolist()
    }
    
    stats_path = 'models/vocabulary_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(vocab_stats, f, indent=2)
    
    print(f"✅ Vocabulary statistics saved: {stats_path}")
    
    # 3. Create example transformation file
    example_texts = [
        "BREAKING: Miracle cure discovered for all diseases!",
        "Scientific study confirms climate change is real.",
        "Secret government documents leaked online.",
        "New economic data shows growth in employment."
    ]
    
    transformed = vectorizer.transform(example_texts)
    
    examples = []
    for i, text in enumerate(example_texts):
        # Get non-zero features for this text
        row = transformed[i]
        indices = row.nonzero()[1]
        values = row.data
        
        # Get feature names and values
        features = []
        for idx, val in zip(indices[:10], values[:10]):  # First 10 features
            feature_name = feature_names[idx]
            features.append({
                'feature': feature_name,
                'value': float(val),
                'type': 'unigram' if ' ' not in feature_name else 'ngram'
            })
        
        examples.append({
            'text': text,
            'total_features': len(indices),
            'top_features': features,
            'vector_shape': transformed.shape[1]
        })
    
    examples_path = 'models/transformation_examples.json'
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"✅ Transformation examples saved: {examples_path}")
    
    return {
        'feature_names': feature_names_path,
        'vocab_stats': stats_path,
        'examples': examples_path
    }

def create_usage_examples():
    """Create usage examples for the vectorizer"""
    
    print("\n" + "=" * 60)
    print("CREATING USAGE EXAMPLES")
    print("=" * 60)
    
    usage_code = '''
"""
vectorizer_usage.py
Example usage of the trained vectorizer
"""

import joblib
import numpy as np

def demonstrate_vectorizer():
    """Demonstrate how to use the vectorizer.pkl file"""
    
    print("=" * 60)
    print("VECTORIZER USAGE DEMONSTRATION")
    print("=" * 60)
    
    # 1. Load the vectorizer
    print("\\n1. Loading vectorizer...")
    try:
        vectorizer = joblib.load('models/vectorizer.pkl')
        print(f"   ✅ Loaded: {type(vectorizer).__name__}")
        
        # Show metadata if available
        if hasattr(vectorizer, 'metadata'):
            print(f"   📊 Metadata:")
            print(f"      Type: {vectorizer.metadata.get('vectorizer_type')}")
            print(f"      Version: {vectorizer.metadata.get('version')}")
            print(f"      Features: {vectorizer.metadata['vocabulary_stats']['total_features']}")
    except Exception as e:
        print(f"   ❌ Error loading vectorizer: {e}")
        return
    
    # 2. Get vocabulary information
    print("\\n2. Vocabulary information:")
    feature_names = vectorizer.get_feature_names_out()
    print(f"   Total features: {len(feature_names)}")
    print(f"   First 5 features: {feature_names[:5]}")
    print(f"   Last 5 features: {feature_names[-5:]}")
    
    # 3. Transform sample texts
    print("\\n3. Transforming sample texts:")
    
    sample_texts = [
        "Breaking news about amazing discovery",
        "Scientific study confirms hypothesis",
        "Secret government documents leaked online",
        "Economic data shows positive growth trends"
    ]
    
    for i, text in enumerate(sample_texts):
        # Transform single text
        transformed = vectorizer.transform([text])
        
        print(f"\\n   Text {i+1}: '{text}'")
        print(f"   Shape: {transformed.shape}")
        print(f"   Non-zero features: {transformed.nnz}")
        
        # Show top features by value
        if transformed.nnz > 0:
            # Get non-zero indices and values
            indices = transformed.nonzero()[1]
            values = transformed.data
            
            # Sort by value (descending)
            sorted_indices = indices[np.argsort(values)[::-1]]
            sorted_values = np.sort(values)[::-1]
            
            # Get top 3 features
            top_features = []
            for idx, val in zip(sorted_indices[:3], sorted_values[:3]):
                feature_name = feature_names[idx]
                top_features.append(f"{feature_name}: {val:.4f}")
            
            print(f"   Top features: {', '.join(top_features)}")
    
    # 4. Batch transformation
    print("\\n4. Batch transformation:")
    batch_transformed = vectorizer.transform(sample_texts)
    print(f"   Batch shape: {batch_transformed.shape}")
    print(f"   Total non-zero elements: {batch_transformed.nnz}")
    print(f"   Sparsity: {100 * (1 - batch_transformed.nnz / batch_transformed.size):.1f}%")
    
    # 5. Feature importance analysis
    print("\\n5. Feature importance example:")
    
    # Simulate model coefficients (in real case, these would come from your model)
    # Let's assume some features are important for fake news detection
    fake_news_indicators = ['breaking', 'secret', 'leaked', 'amazing', 'discovery']
    
    print("   Fake news indicator features:")
    for indicator in fake_news_indicators:
        # Find features containing this indicator
        matching_features = [f for f in feature_names if indicator in f]
        if matching_features:
            print(f"      '{indicator}': {len(matching_features)} features")
            if len(matching_features) <= 3:
                print(f"         {matching_features}")
    
    print("\\n" + "=" * 60)
    print("USAGE EXAMPLES:")
    print("=" * 60)
    
    print('''
# Example 1: Basic transformation
vectorizer = joblib.load('models/vectorizer.pkl')
features = vectorizer.transform(["Your text here"])
print(f"Features shape: {features.shape}")

# Example 2: Getting feature names
feature_names = vectorizer.get_feature_names_out()
print(f"Total features: {len(feature_names)}")

# Example 3: With model prediction
model = joblib.load('models/fake_news_model.pkl')
text = "Breaking news about secret discovery"
features = vectorizer.transform([text])
prediction = model.predict(features)
probability = model.predict_proba(features)
print(f"Prediction: {prediction[0]}, Probability: {probability[0]}")

# Example 4: Batch processing
texts = ["text1", "text2", "text3"]
batch_features = vectorizer.transform(texts)
# Use with model: model.predict(batch_features)
    ''')
    
    return True

if __name__ == "__main__":
    demonstrate_vectorizer()
'''
    
    usage_path = 'vectorizer_usage.py'
    with open(usage_path, 'w') as f:
        f.write(usage_code)
    
    print(f"✅ Usage examples saved: {usage_path}")
    
    return usage_path

def main():
    """Main function to create everything"""
    
    print("\n" + "=" * 70)
    print("COMPLETE VECTORIZER.PKL CREATION")
    print("=" * 70)
    
    # Create the main vectorizer
    vectorizer_path, vectorizer = create_vectorizer_pkl()
    
    # Create advanced vectorizer (optional)
    advanced_path = create_advanced_vectorizer()
    
    # Create supporting files
    supporting_files = create_supporting_files()
    
    # Create usage examples
    usage_path = create_usage_examples()
    
    print("\n" + "=" * 70)
    print("✅ CREATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n📁 Files created:")
    print(f"   1. {vectorizer_path} - Main TF-IDF vectorizer")
    print(f"   2. {advanced_path} - Advanced vectorizer (optional)")
    print(f"   3. {supporting_files['feature_names']} - All feature names")
    print(f"   4. {supporting_files['vocab_stats']} - Vocabulary statistics")
    print(f"   5. {supporting_files['examples']} - Transformation examples")
    print(f"   6. {usage_path} - Usage examples")
    
    print(f"\n🚀 Quick test:")
    print(f"   python vectorizer_usage.py")
    
    print(f"\n📊 Vectorizer information:")
    print(f"   - Type: TF-IDF Vectorizer")
    print(f"   - Features: {len(vectorizer.get_feature_names_out())}")
    print(f"   - N-gram range: {vectorizer.ngram_range}")
    print(f"   - Max features: {vectorizer.max_features}")
    
    print(f"\n💡 How to use with your model:")
    print(f'''
# Load both model and vectorizer
import joblib

model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Process new text
new_text = "Breaking news about secret discovery"
features = vectorizer.transform([new_text])
prediction = model.predict(features)
probability = model.predict_proba(features)

print(f"Prediction: {prediction[0]}")
print(f"Probability - Real: {probability[0][0]:.2%}")
print(f"Probability - Fake: {probability[0][1]:.2%}")
    ''')
    
    return True

if __name__ == "__main__":
    main()
