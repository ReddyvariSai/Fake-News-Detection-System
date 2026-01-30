import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List, Dict, Any, Optional, Union
import emoji
import html

class FakeNewsPreprocessor:
    """
    Advanced text preprocessor specifically designed for fake news detection
    """
    
    def __init__(self, language='english', use_stemming=False, 
                 keep_punctuation=False, min_word_length=2):
        """
        Initialize the preprocessor
        
        Parameters:
        -----------
        language : str
            Language for text processing
        use_stemming : bool
            Use stemming instead of lemmatization
        keep_punctuation : bool
            Whether to keep basic punctuation
        min_word_length : int
            Minimum word length to keep
        """
        self.language = language
        self.use_stemming = use_stemming
        self.keep_punctuation = keep_punctuation
        self.min_word_length = min_word_length
        
        # Download required NLTK data
        self._download_nltk_resources()
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Fake news specific stopwords (words that are too common in both real/fake news)
        self.fake_news_stopwords = {
            'news', 'article', 'report', 'said', 'says', 'according',
            'would', 'could', 'also', 'like', 'get', 'go', 'know',
            'think', 'make', 'take', 'see', 'come', 'want', 'look',
            'use', 'find', 'give', 'tell', 'new', 'good', 'well',
            'first', 'last', 'one', 'two', 'three', 'time', 'people',
            'year', 'day', 'week', 'month', 'today', 'yesterday'
        }
        self.stop_words.update(self.fake_news_stopwords)
        
        # Sensational words to flag (but not remove)
        self.sensational_words = {
            'breaking', 'shocking', 'secret', 'amazing', 'unbelievable',
            'hidden', 'cover-up', 'exposed', 'scandal', 'leaked', 'urgent',
            'emergency', 'warning', 'alert', 'dangerous', 'explosive',
            'miracle', 'instant', 'guaranteed', 'classified', 'forbidden',
            'banned', 'controversial', 'sensational', 'bombshell'
        }
        
        # Conspiracy theory indicators
        self.conspiracy_indicators = {
            'conspiracy', 'deep state', 'new world order', 'illuminati',
            'false flag', 'hoax', 'mainstream media', 'establishment',
            'elite', 'globalists', 'shadow government', 'cover up'
        }
        
        # Credibility indicators
        self.credibility_indicators = {
            'according to study', 'research shows', 'scientists found',
            'peer-reviewed', 'published in', 'data indicates', 'statistics show',
            'expert said', 'official report', 'clinical trial', 'meta-analysis'
        }
        
        # Emotional words
        self.emotional_words = {
            'outrageous', 'horrifying', 'terrifying', 'disgusting',
            'frightening', 'appalling', 'shameful', 'scandalous'
        }
        
        # Initialize statistics
        self.stats = {
            'total_texts_processed': 0,
            'total_words_removed': 0,
            'total_stopwords_removed': 0,
            'avg_text_length_before': 0,
            'avg_text_length_after': 0
        }
        
        # Metadata
        self.metadata = {
            'created_date': datetime.now().isoformat(),
            'version': '2.0.0',
            'language': language,
            'use_stemming': use_stemming,
            'keep_punctuation': keep_punctuation,
            'min_word_length': min_word_length,
            'stopwords_count': len(self.stop_words),
            'sensational_words_count': len(self.sensational_words),
            'description': 'Advanced text preprocessor for fake news detection'
        }
    
    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        required_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
            except LookupError:
                print(f"Downloading NLTK resource: {package}")
                nltk.download(package, quiet=True)
    
    def clean_text(self, text: str, remove_urls=True, remove_emails=True,
                  remove_html=True, remove_emoji=False, 
                  expand_contractions=True) -> str:
        """
        Clean text by removing unwanted elements
        
        Parameters:
        -----------
        text : str
            Input text
        remove_urls : bool
            Remove URLs
        remove_emails : bool
            Remove email addresses
        remove_html : bool
            Remove HTML tags
        remove_emoji : bool
            Remove emojis
        expand_contractions : bool
            Expand contractions (don't -> do not)
            
        Returns:
        --------
        str : Cleaned text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML entities
        if remove_html:
            text = html.unescape(text)
            text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
        
        # Remove email addresses
        if remove_emails:
            text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Remove emojis if requested
        if remove_emoji:
            text = emoji.replace_emoji(text, replace='')
        
        # Expand contractions
        if expand_contractions:
            contractions = {
                "don't": "do not",
                "can't": "cannot",
                "won't": "will not",
                "it's": "it is",
                "they're": "they are",
                "we're": "we are",
                "you're": "you are",
                "I'm": "I am",
                "isn't": "is not",
                "aren't": "are not",
                "wasn't": "was not",
                "weren't": "were not",
                "haven't": "have not",
                "hasn't": "has not",
                "hadn't": "had not",
                "doesn't": "does not",
                "don't": "do not",
                "didn't": "did not",
                "couldn't": "could not",
                "shouldn't": "should not",
                "wouldn't": "would not",
                "mustn't": "must not",
                "mightn't": "might not",
                "needn't": "need not"
            }
            
            for contraction, expansion in contractions.items():
                text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_text(self, text: str, to_lowercase=True, 
                      remove_numbers=True, remove_special_chars=True,
                      keep_basic_punct=False) -> str:
        """
        Normalize text by standardizing formatting
        
        Parameters:
        -----------
        text : str
            Input text
        to_lowercase : bool
            Convert to lowercase
        remove_numbers : bool
            Remove numbers
        remove_special_chars : bool
            Remove special characters
        keep_basic_punct : bool
            Keep basic punctuation (.!?,)
            
        Returns:
        --------
        str : Normalized text
        """
        # Convert to lowercase
        if to_lowercase:
            text = text.lower()
        
        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove special characters
        if remove_special_chars:
            if keep_basic_punct:
                # Keep basic punctuation
                text = re.sub(r'[^\w\s.,!?-]', '', text)
            else:
                # Remove all punctuation
                text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def tokenize_text(self, text: str, return_tokens=True) -> Union[str, List[str]]:
        """
        Tokenize and process text
        
        Parameters:
        -----------
        text : str
            Input text
        return_tokens : bool
            Return tokens list instead of joined string
            
        Returns:
        --------
        Union[str, List[str]]: Processed text or tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Process tokens
        processed_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_word_length:
                continue
            
            # Remove stopwords
            if token in self.stop_words:
                self.stats['total_stopwords_removed'] += 1
                continue
            
            # Apply stemming or lemmatization
            if self.use_stemming:
                token = self.stemmer.stem(token)
            else:
                token = self.lemmatizer.lemmatize(token)
            
            processed_tokens.append(token)
        
        self.stats['total_words_removed'] += (len(tokens) - len(processed_tokens))
        
        if return_tokens:
            return processed_tokens
        else:
            return ' '.join(processed_tokens)
    
    def analyze_text_features(self, text: str) -> Dict[str, Any]:
        """
        Extract advanced text features for fake news analysis
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        dict : Text analysis features
        """
        text_lower = text.lower()
        
        features = {
            # Basic statistics
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            
            # Punctuation analysis
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'all_caps_words': len([word for word in text.split() if word.isupper() and len(word) > 1]),
            
            # Fake news indicators
            'sensational_score': sum(1 for word in self.sensational_words if word in text_lower),
            'conspiracy_score': sum(1 for term in self.conspiracy_indicators if term in text_lower),
            'emotional_score': sum(1 for word in self.emotional_words if word in text_lower),
            'credibility_score': sum(1 for term in self.credibility_indicators if term in text_lower),
            
            # Word statistics
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'max_word_length': max([len(word) for word in text.split()]) if text.split() else 0,
            'unique_word_ratio': len(set(text_lower.split())) / max(len(text_lower.split()), 1),
            
            # Special patterns
            'has_breaking': 'breaking' in text_lower,
            'has_exclusive': 'exclusive' in text_lower,
            'has_urgent': 'urgent' in text_lower,
            'has_warning': 'warning' in text_lower,
            'has_secret': 'secret' in text_lower,
            
            # Quote analysis
            'quote_count': text.count('"') + text.count("'"),
            'has_anonymous': 'anonymous' in text_lower,
            'has_source': 'source' in text_lower or 'sources' in text_lower,
            
            # Readability indicators
            'avg_words_per_sentence': len(text.split()) / max(len(sent_tokenize(text)), 1),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_percentages': bool(re.search(r'\d+%', text)),
            
            # Modal verbs (common in speculative/fake news)
            'modal_verb_count': sum(text_lower.count(verb) for verb in 
                                   ['might', 'could', 'would', 'may', 'can', 'should']),
            
            # First person pronouns (common in opinion pieces)
            'first_person_count': sum(text_lower.count(pronoun) for pronoun in 
                                     ['i', 'we', 'our', 'us', 'my', 'mine'])
        }
        
        # Calculate composite scores
        features['fake_news_likelihood'] = (
            features['sensational_score'] * 0.3 +
            features['conspiracy_score'] * 0.3 +
            features['emotional_score'] * 0.2 +
            features['exclamation_count'] * 0.1 +
            features['all_caps_words'] * 0.1
        ) / 10  # Normalize to 0-1
        
        features['credibility_score_norm'] = features['credibility_score'] / max(features['word_count'], 1)
        
        return features
    
    def preprocess_pipeline(self, text: str, return_analysis=False, 
                          return_tokens=False) -> Union[str, List[str], Dict[str, Any]]:
        """
        Complete preprocessing pipeline
        
        Parameters:
        -----------
        text : str
            Input text
        return_analysis : bool
            Return text analysis features
        return_tokens : bool
            Return tokens instead of joined text
            
        Returns:
        --------
        Union[str, List[str], Dict[str, Any]]: Processed text, tokens, or analysis
        """
        # Update statistics
        self.stats['total_texts_processed'] += 1
        original_length = len(text)
        
        # Step 1: Clean text
        cleaned_text = self.clean_text(text)
        
        # Step 2: Normalize text
        normalized_text = self.normalize_text(
            cleaned_text, 
            keep_basic_punct=self.keep_punctuation
        )
        
        # Step 3: Tokenize and process
        if return_tokens:
            result = self.tokenize_text(normalized_text, return_tokens=True)
        else:
            result = self.tokenize_text(normalized_text, return_tokens=False)
        
        # Update length statistics
        processed_length = len(result) if isinstance(result, str) else sum(len(t) for t in result)
        self.stats['avg_text_length_before'] = (
            self.stats['avg_text_length_before'] * (self.stats['total_texts_processed'] - 1) + 
            original_length
        ) / self.stats['total_texts_processed']
        self.stats['avg_text_length_after'] = (
            self.stats['avg_text_length_after'] * (self.stats['total_texts_processed'] - 1) + 
            processed_length
        ) / self.stats['total_texts_processed']
        
        if return_analysis:
            analysis = self.analyze_text_features(text)
            analysis['processed_text'] = result if isinstance(result, str) else ' '.join(result)
            return analysis
        else:
            return result
    
    def batch_preprocess(self, texts: List[str], show_progress=False) -> List[str]:
        """
        Preprocess a batch of texts
        
        Parameters:
        -----------
        texts : list
            List of texts to preprocess
        show_progress : bool
            Show progress bar
            
        Returns:
        --------
        list : List of preprocessed texts
        """
        processed_texts = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Preprocessing texts")
        else:
            iterator = texts
        
        for text in iterator:
            processed = self.preprocess_pipeline(text)
            processed_texts.append(processed)
        
        return processed_texts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        stats = self.stats.copy()
        stats['compression_ratio'] = (
            stats['avg_text_length_after'] / stats['avg_text_length_before'] 
            if stats['avg_text_length_before'] > 0 else 0
        )
        return stats
    
    def save(self, filepath: str):
        """Save the preprocessor to file"""
        joblib.dump(self, filepath, compress=3)
        print(f"✅ Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FakeNewsPreprocessor':
        """Load a preprocessor from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        preprocessor = joblib.load(filepath)
        print(f"✅ Preprocessor loaded from {filepath}")
        print(f"   Version: {preprocessor.metadata.get('version')}")
        print(f"   Language: {preprocessor.metadata.get('language')}")
        print(f"   Texts processed: {preprocessor.stats.get('total_texts_processed', 0)}")
        
        return preprocessor
    
    def __repr__(self):
        return (f"FakeNewsPreprocessor(language='{self.language}', "
                f"version={self.metadata.get('version')}, "
                f"texts_processed={self.stats['total_texts_processed']})")

def create_and_save_preprocessor():
    """Create and save a preprocessor instance"""
    
    print("=" * 70)
    print("CREATING FAKE NEWS TEXT PREPROCESSOR")
    print("=" * 70)
    
    # Create preprocessor
    print("\n🔧 Creating preprocessor...")
    preprocessor = FakeNewsPreprocessor(
        language='english',
        use_stemming=False,  # Use lemmatization (more accurate)
        keep_punctuation=False,
        min_word_length=2
    )
    
    print(f"✅ Preprocessor created:")
    print(f"   Language: {preprocessor.language}")
    print(f"   Stopwords: {len(preprocessor.stop_words)}")
    print(f"   Sensational words: {len(preprocessor.sensational_words)}")
    print(f"   Version: {preprocessor.metadata['version']}")
    
    # Test with sample texts
    print("\n🧪 Testing preprocessor...")
    
    test_texts = [
        # Real news example
        "Scientists at Harvard University have published a new study in Nature journal showing that climate change is accelerating. The research, which involved data from 50 countries, indicates a 2°C temperature rise over the past century.",
        
        # Fake news example
        "BREAKING: SHOCKING discovery! Secret government lab creates MIRACLE cure that ELIMINATES all diseases overnight! Doctors are FURIOUS - Big Pharma is trying to SUPPRESS this information! Share NOW before it's DELETED!",
        
        # Mixed example
        "According to official data released by the CDC, vaccination rates have increased by 15% over the past month. Experts say this is encouraging news for public health.",
        
        # Sensational example
        "URGENT WARNING: They're putting DANGEROUS CHEMICALS in our food supply! The TRUTH about what's REALLY in your groceries will SHOCK you!",
        
        # Simple example
        "This is a test article about fake news detection using machine learning."
    ]
    
    for i, text in enumerate(test_texts[:3], 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}:")
        print(f"Original ({len(text)} chars):")
        print(f"  '{text[:80]}...'" if len(text) > 80 else f"  '{text}'")
        
        # Preprocess
        processed = preprocessor.preprocess_pipeline(text)
        print(f"\nProcessed ({len(processed)} chars):")
        print(f"  '{processed[:80]}...'" if len(processed) > 80 else f"  '{processed}'")
        
        # Get analysis
        analysis = preprocessor.analyze_text_features(text)
        print(f"\nAnalysis:")
        print(f"  Words: {analysis['word_count']}")
        print(f"  Sentences: {analysis['sentence_count']}")
        print(f"  Exclamation marks: {analysis['exclamation_count']}")
        print(f"  Sensational score: {analysis['sensational_score']}")
        print(f"  Credibility score: {analysis['credibility_score']}")
        print(f"  Fake news likelihood: {analysis['fake_news_likelihood']:.2%}")
    
    # Batch processing test
    print(f"\n{'─' * 60}")
    print("Batch processing test...")
    batch_results = preprocessor.batch_preprocess(test_texts)
    print(f"✅ Processed {len(batch_results)} texts")
    print(f"  Original lengths: {[len(t) for t in test_texts]}")
    print(f"  Processed lengths: {[len(r) for r in batch_results]}")
    
    # Save the preprocessor
    print("\n💾 Saving preprocessor...")
    os.makedirs('models', exist_ok=True)
    preprocessor_path = 'models/preprocessor.pkl'
    preprocessor.save(preprocessor_path)
    
    # Verify file creation
    file_size = os.path.getsize(preprocessor_path)
    print(f"✅ Preprocessor saved successfully!")
    print(f"   File: {preprocessor_path}")
    print(f"   Size: {file_size:,} bytes")
    
    # Test loading back
    print("\n🔄 Testing load functionality...")
    loaded_preprocessor = FakeNewsPreprocessor.load(preprocessor_path)
    
    # Test that loaded preprocessor works
    test_text = "This is a test to verify the loaded preprocessor works correctly."
    processed = loaded_preprocessor.preprocess_pipeline(test_text)
    print(f"✅ Loaded preprocessor test:")
    print(f"   Input: '{test_text}'")
    print(f"   Output: '{processed}'")
    
    # Show statistics
    print(f"\n📊 Preprocessor statistics:")
    stats = preprocessor.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    return preprocessor_path, preprocessor

def create_supporting_examples():
    """Create supporting examples and documentation"""
    
    print("\n" + "=" * 70)
    print("CREATING SUPPORTING EXAMPLES")
    print("=" * 70)
    
    # Create usage example
    usage_code = '''
"""
preprocessor_usage.py
Example usage of the FakeNewsPreprocessor
"""

import joblib

def demonstrate_preprocessor():
    """Demonstrate how to use the preprocessor"""
    
    print("=" * 60)
    print("FAKE NEWS PREPROCESSOR USAGE")
    print("=" * 60)
    
    # 1. Load the preprocessor
    print("\\n1. Loading preprocessor...")
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        print(f"   ✅ Loaded: {preprocessor}")
        print(f"   📊 Texts processed: {preprocessor.stats['total_texts_processed']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # 2. Basic preprocessing
    print("\\n2. Basic preprocessing:")
    texts = [
        "Scientists discover NEW evidence of climate change!",
        "BREAKING: Secret government documents LEAKED online!",
        "Economic data shows steady growth according to experts."
    ]
    
    for i, text in enumerate(texts, 1):
        processed = preprocessor.preprocess_pipeline(text)
        print(f"\\n   Example {i}:")
        print(f"   Original: {text}")
        print(f"   Processed: {processed}")
    
    # 3. Text analysis
    print("\\n3. Text analysis features:")
    sample_text = "BREAKING: SHOCKING discovery! Secret cure found for all diseases!"
    analysis = preprocessor.analyze_text_features(sample_text)
    
    print(f"\\n   Text: '{sample_text}'")
    print(f"   Word count: {analysis['word_count']}")
    print(f"   Exclamation marks: {analysis['exclamation_count']}")
    print(f"   Sensational score: {analysis['sensational_score']}")
    print(f"   Fake news likelihood: {analysis['fake_news_likelihood']:.1%}")
    
    # 4. Batch processing
    print("\\n4. Batch processing:")
    batch_texts = [
        "First sample text for processing.",
        "Second example with different content.",
        "Third text to demonstrate batch capabilities."
    ]
    
    batch_results = preprocessor.batch_preprocess(batch_texts, show_progress=False)
    print(f"   Processed {len(batch_results)} texts")
    
    # 5. Integration with vectorizer and model
    print("\\n5. Complete pipeline example:")
    
    pipeline_example = '''
# Complete fake news detection pipeline
import joblib

# Load all components
preprocessor = joblib.load('models/preprocessor.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
model = joblib.load('models/fake_news_model.pkl')

def predict_news(text):
    """Predict if news is real or fake"""
    
    # 1. Preprocess text
    processed_text = preprocessor.preprocess_pipeline(text)
    
    # 2. Vectorize (convert to features)
    features = vectorizer.transform([processed_text])
    
    # 3. Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    # 4. Analyze text features
    analysis = preprocessor.analyze_text_features(text)
    
    return {
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'confidence': probability[1] if prediction == 1 else probability[0],
        'analysis': {
            'sensational_score': analysis['sensational_score'],
            'credibility_score': analysis['credibility_score'],
            'fake_news_likelihood': analysis['fake_news_likelihood']
        }
    }

# Example usage
news_text = "BREAKING: Amazing discovery cures all diseases!"
result = predict_news(news_text)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
    '''
    
    print(pipeline_example)
    
    print("\\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_preprocessor()
'''
    
    # Save usage example
    usage_path = 'preprocessor_usage.py'
    with open(usage_path, 'w') as f:
        f.write(usage_code)
    
    print(f"✅ Created usage example: {usage_path}")
    
    # Create test script
    test_code = '''
"""
test_preprocessor.py
Test the preprocessor with various text examples
"""

import joblib
import pandas as pd

def test_preprocessor():
    """Test the preprocessor with different types of text"""
    
    # Load preprocessor
    print("Loading preprocessor...")
    preprocessor = joblib.load('models/preprocessor.pkl')
    
    # Test cases
    test_cases = [
        {
            "name": "Real News",
            "text": "A new study published in the Journal of Medicine shows that regular exercise reduces the risk of heart disease by 30%. The research involved 10,000 participants over 5 years.",
            "expected_type": "real"
        },
        {
            "name": "Fake News (Sensational)",
            "text": "BREAKING: SHOCKING discovery! One simple herb CURES cancer overnight! Doctors are FURIOUS! Big Pharma doesn't want you to know!",
            "expected_type": "fake"
        },
        {
            "name": "Government Report",
            "text": "According to data released by the Bureau of Labor Statistics, unemployment fell to 3.5% last month, indicating continued economic growth.",
            "expected_type": "real"
        },
        {
            "name": "Conspiracy Theory",
            "text": "SECRET government documents reveal shocking truth about alien technology! The deep state is covering it up! Share this before they delete it!",
            "expected_type": "fake"
        },
        {
            "name": "Scientific Abstract",
            "text": "Our meta-analysis of 25 clinical trials demonstrates the efficacy of the new treatment with a p-value of 0.001. Results were consistent across demographic groups.",
            "expected_type": "real"
        },
        {
            "name": "Clickbait Headline",
            "text": "You won't BELIEVE what this celebrity said about vaccines! The truth will SHOCK you!",
            "expected_type": "fake"
        }
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\\n{'─' * 60}")
        print(f"Test: {test['name']}")
        print(f"Text: {test['text'][:80]}...")
        
        # Process text
        processed = preprocessor.preprocess_pipeline(test['text'])
        
        # Analyze text
        analysis = preprocessor.analyze_text_features(test['text'])
        
        # Determine if text shows fake news characteristics
        is_suspicious = (
            analysis['sensational_score'] > 2 or
            analysis['conspiracy_score'] > 0 or
            analysis['exclamation_count'] > 2 or
            analysis['fake_news_likelihood'] > 0.5
        )
        
        detected_type = "fake" if is_suspicious else "real"
        correct = detected_type == test['expected_type']
        
        results.append({
            "Test": test['name'],
            "Expected": test['expected_type'],
            "Detected": detected_type,
            "Correct": "✅" if correct else "❌",
            "Sensational Score": analysis['sensational_score'],
            "Credibility Score": analysis['credibility_score'],
            "Fake News Likelihood": f"{analysis['fake_news_likelihood']:.1%}",
            "Exclamation Marks": analysis['exclamation_count']
        })
        
        print(f"Processed: {processed[:80]}...")
        print(f"Sensational score: {analysis['sensational_score']}")
        print(f"Credibility score: {analysis['credibility_score']}")
        print(f"Fake news likelihood: {analysis['fake_news_likelihood']:.1%}")
        print(f"Detection: {detected_type.upper()} (Expected: {test['expected_type'].upper()})")
    
    # Print summary
    print(f"\\n{'=' * 60}")
    print("TEST SUMMARY")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    accuracy = sum(1 for r in results if r['Correct'] == '✅') / len(results)
    print(f"\\n📊 Accuracy: {accuracy:.1%} ({sum(1 for r in results if r['Correct'] == '✅')}/{len(results)})")
    
    return accuracy > 0.7

if __name__ == "__main__":
    success = test_preprocessor()
    if success:
        print("\\n🎉 Preprocessor test PASSED!")
    else:
        print("\\n⚠️ Preprocessor test needs improvement.")
'''
    
    test_path = 'test_preprocessor.py'
    with open(test_path, 'w') as f:
        f.write(test_code)
    
    print(f"✅ Created test script: {test_path}")
    
    return usage_path, test_path

def main():
    """Main function to create everything"""
    
    print("\n" + "=" * 80)
    print("COMPLETE FAKE NEWS PREPROCESSOR CREATION")
    print("=" * 80)
    
    # Create and save preprocessor
    preprocessor_path, preprocessor = create_and_save_preprocessor()
    
    # Create supporting examples
    usage_path, test_path = create_supporting_examples()
    
    print("\n" + "=" * 80)
    print("✅ CREATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n📁 Files created:")
    print(f"   1. {preprocessor_path} - Main preprocessor")
    print(f"   2. {usage_path} - Usage examples")
    print(f"   3. {test_path} - Test script")
    
    print(f"\n🚀 Quick start:")
    print(f"   python preprocessor_usage.py")
    print(f"   python test_preprocessor.py")
    
    print(f"\n📊 Preprocessor capabilities:")
    print(f"   • Text cleaning (URLs, HTML, emails)")
    print(f"   • Normalization (lowercase, punctuation)")
    print(f"   • Tokenization and lemmatization")
    print(f"   • Stopword removal ({len(preprocessor.stop_words)} stopwords)")
    print(f"   • Fake news feature analysis")
    print(f"   • Batch processing support")
    
    print(f"\n💡 How to use in your project:")
    print(f'''
import joblib

# Load the preprocessor
preprocessor = joblib.load('models/preprocessor.pkl')

# Process single text
text = "Your news article here..."
processed = preprocessor.preprocess_pipeline(text)

# Get text analysis
analysis = preprocessor.analyze_text_features(text)
print(f"Sensational score: {analysis['sensational_score']}")
print(f"Fake news likelihood: {analysis['fake_news_likelihood']:.1%}")

# Batch processing
texts = ["text1", "text2", "text3"]
processed_batch = preprocessor.batch_preprocess(texts)
    ''')
    
    print(f"\n🔍 Fake news detection features:")
    print(f"   • Sensational word detection: {len(preprocessor.sensational_words)} words")
    print(f"   • Conspiracy theory indicators: {len(preprocessor.conspiracy_indicators)}")
    print(f"   • Credibility indicators: {len(preprocessor.credibility_indicators)}")
    print(f"   • Emotional language detection: {len(preprocessor.emotional_words)} words")
    
    return True

if __name__ == "__main__":
    main()
