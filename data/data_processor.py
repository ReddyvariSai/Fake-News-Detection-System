import pandas as pd
import numpy as np
import re
import string
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import html
import unicodedata

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class DataProcessor:
    """Advanced data processing pipeline for fake news detection"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self._setup_special_characters()
    
    def _get_default_config(self) -> Dict:
        """Get default processing configuration"""
        return {
            "text_cleaning": {
                "remove_html": True,
                "remove_urls": True,
                "remove_mentions": True,
                "remove_hashtags": True,
                "remove_numbers": False,
                "remove_special_chars": True,
                "remove_extra_spaces": True,
                "lowercase": True,
                "remove_stopwords": True,
                "stem": True,
                "lemmatize": False,
                "min_word_length": 2,
                "max_word_length": 20,
                "remove_emojis": True,
                "normalize_unicode": True
            },
            "feature_engineering": {
                "text_features": True,
                "linguistic_features": True,
                "readability_features": True,
                "sentiment_features": True,
                "metadata_features": True,
                "domain_features": True
            }
        }
    
    def _setup_special_characters(self):
        """Setup special character patterns"""
        # Emoji pattern
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        # URL pattern
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        
        # Mention pattern
        self.mention_pattern = re.compile(r'@\w+')
        
        # Hashtag pattern
        self.hashtag_pattern = re.compile(r'#\w+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean text with multiple options
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        text = str(text).strip()
        
        # Remove HTML entities
        if self.config["text_cleaning"].get("remove_html", True):
            text = html.unescape(text)
        
        # Normalize Unicode
        if self.config["text_cleaning"].get("normalize_unicode", True):
            text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs
        if self.config["text_cleaning"].get("remove_urls", True):
            text = self.url_pattern.sub('', text)
        
        # Remove mentions
        if self.config["text_cleaning"].get("remove_mentions", True):
            text = self.mention_pattern.sub('', text)
        
        # Remove hashtags
        if self.config["text_cleaning"].get("remove_hashtags", True):
            text = self.hashtag_pattern.sub('', text)
        
        # Remove emojis
        if self.config["text_cleaning"].get("remove_emojis", True):
            text = self.emoji_pattern.sub('', text)
        
        # Remove numbers
        if self.config["text_cleaning"].get("remove_numbers", True):
            text = re.sub(r'\d+', '', text)
        
        # Remove special characters
        if self.config["text_cleaning"].get("remove_special_chars", True):
            # Keep alphanumeric, spaces, and basic punctuation
            text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Convert to lowercase
        if self.config["text_cleaning"].get("lowercase", True):
            text = text.lower()
        
        # Remove extra spaces
        if self.config["text_cleaning"].get("remove_extra_spaces", True):
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Filter by word length
        min_len = self.config["text_cleaning"].get("min_word_length", 2)
        max_len = self.config["text_cleaning"].get("max_word_length", 20)
        tokens = [token for token in tokens if min_len <= len(token) <= max_len]
        
        # Remove stopwords
        if self.config["text_cleaning"].get("remove_stopwords", True):
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming
        if self.config["text_cleaning"].get("stem", True):
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization (if enabled and not stemming)
        if (self.config["text_cleaning"].get("lemmatize", False) and 
            not self.config["text_cleaning"].get("stem", True)):
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def extract_text_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Extract comprehensive text-based features
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
        
        Returns:
            DataFrame with extracted features
        """
        features_df = pd.DataFrame(index=df.index)
        
        # Basic text statistics
        features_df['text_length'] = df[text_column].str.len()
        features_df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
        features_df['char_count'] = df[text_column].str.len()
        features_df['avg_word_length'] = df[text_column].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).strip() else 0
        )
        
        # Sentence statistics (simplified)
        features_df['sentence_count'] = df[text_column].apply(
            lambda x: len(re.split(r'[.!?]+', str(x)))
        )
        features_df['avg_sentence_length'] = (
            features_df['word_count'] / features_df['sentence_count']
        ).replace([np.inf, -np.inf], 0)
        
        # Vocabulary richness
        features_df['unique_word_count'] = df[text_column].apply(
            lambda x: len(set(str(x).lower().split()))
        )
        features_df['lexical_diversity'] = (
            features_df['unique_word_count'] / features_df['word_count']
        ).replace([np.inf, -np.inf], 0)
        
        # Punctuation features
        features_df['exclamation_count'] = df[text_column].str.count('!')
        features_df['question_count'] = df[text_column].str.count('\?')
        features_df['period_count'] = df[text_column].str.count('\.')
        features_df['comma_count'] = df[text_column].str.count(',')
        
        # Capitalization features
        features_df['uppercase_count'] = df[text_column].apply(
            lambda x: sum(1 for c in str(x) if c.isupper())
        )
        features_df['uppercase_ratio'] = (
            features_df['uppercase_count'] / features_df['char_count']
        ).replace([np.inf, -np.inf], 0)
        
        # Special character features
        features_df['digit_count'] = df[text_column].apply(
            lambda x: sum(1 for c in str(x) if c.isdigit())
        )
        features_df['special_char_count'] = df[text_column].apply(
            lambda x: sum(1 for c in str(x) if not c.isalnum() and not c.isspace())
        )
        
        # Readability scores (simplified)
        features_df['flesch_reading_ease'] = df[text_column].apply(self._calculate_flesch_reading_ease)
        features_df['smog_index'] = df[text_column].apply(self._calculate_smog_index)
        
        # Sentiment features (using simple word lists)
        features_df['positive_word_count'] = df[text_column].apply(self._count_positive_words)
        features_df['negative_word_count'] = df[text_column].apply(self._count_negative_words)
        features_df['sentiment_score'] = (
            features_df['positive_word_count'] - features_df['negative_word_count']
        )
        
        # Text complexity features
        features_df['long_word_count'] = df[text_column].apply(
            lambda x: sum(1 for word in str(x).split() if len(word) > 6)
        )
        features_df['long_word_ratio'] = (
            features_df['long_word_count'] / features_df['word_count']
        ).replace([np.inf, -np.inf], 0)
        
        logger.info(f"Extracted {len(features_df.columns)} text features")
        return features_df
    
    def extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract metadata-based features
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with metadata features
        """
        features_df = pd.DataFrame(index=df.index)
        
        # Date features (if available)
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                features_df[f'{date_col}_year'] = df[date_col].dt.year
                features_df[f'{date_col}_month'] = df[date_col].dt.month
                features_df[f'{date_col}_day'] = df[date_col].dt.day
                features_df[f'{date_col}_weekday'] = df[date_col].dt.weekday
                features_df[f'{date_col}_hour'] = df[date_col].dt.hour
            except:
                pass
        
        # Source/domain features
        source_columns = [col for col in df.columns if 'source' in col.lower() or 'domain' in col.lower()]
        for source_col in source_columns:
            if source_col in df.columns:
                # One-hot encoding for top sources
                top_sources = df[source_col].value_counts().head(10).index
                for source in top_sources:
                    features_df[f'source_{source}'] = (df[source_col] == source).astype(int)
        
        # Author features
        author_columns = [col for col in df.columns if 'author' in col.lower()]
        for author_col in author_columns:
            if author_col in df.columns:
                # Author frequency
                author_counts = df[author_col].value_counts()
                features_df[f'{author_col}_frequency'] = df[author_col].map(author_counts)
        
        logger.info(f"Extracted {len(features_df.columns)} metadata features")
        return features_df
    
    def extract_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract domain-specific features
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with domain features
        """
        features_df = pd.DataFrame(index=df.index)
        
        # Load credible/questionable domain lists
        credible_domains = self._load_domain_list('credible_domains.txt')
        questionable_domains = self._load_domain_list('questionable_domains.txt')
        
        # Check if source column exists
        source_cols = [col for col in df.columns if 'source' in col.lower() or 'domain' in col.lower()]
        
        for source_col in source_cols:
            if source_col in df.columns:
                # Domain credibility features
                features_df[f'{source_col}_is_credible'] = df[source_col].apply(
                    lambda x: 1 if any(domain in str(x).lower() for domain in credible_domains) else 0
                )
                features_df[f'{source_col}_is_questionable'] = df[source_col].apply(
                    lambda x: 1 if any(domain in str(x).lower() for domain in questionable_domains) else 0
                )
                features_df[f'{source_col}_is_unknown'] = (
                    (features_df[f'{source_col}_is_credible'] == 0) & 
                    (features_df[f'{source_col}_is_questionable'] == 0)
                ).astype(int)
        
        logger.info(f"Extracted {len(features_df.columns)} domain features")
        return features_df
    
    def process_dataset(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        label_column: Optional[str] = None,
        save_intermediate: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Complete dataset processing pipeline
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column (optional)
            save_intermediate: Whether to save intermediate results
        
        Returns:
            Tuple of (processed features, labels)
        """
        logger.info(f"Starting data processing pipeline for {len(df)} records")
        
        # Clean text column
        logger.info("Cleaning text...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Tokenize
        logger.info("Tokenizing text...")
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_text)
        df['processed_text'] = df['tokens'].apply(' '.join)
        
        # Extract features based on configuration
        feature_config = self.config.get('feature_engineering', {})
        all_features = []
        
        # Text features
        if feature_config.get('text_features', True):
            logger.info("Extracting text features...")
            text_features = self.extract_text_features(df, 'cleaned_text')
            all_features.append(text_features)
        
        # Linguistic features (additional to text features)
        if feature_config.get('linguistic_features', True):
            logger.info("Extracting linguistic features...")
            # Could add more linguistic features here
        
        # Metadata features
        if feature_config.get('metadata_features', True):
            logger.info("Extracting metadata features...")
            metadata_features = self.extract_metadata_features(df)
            all_features.append(metadata_features)
        
        # Domain features
        if feature_config.get('domain_features', True):
            logger.info("Extracting domain features...")
            domain_features = self.extract_domain_features(df)
            all_features.append(domain_features)
        
        # Combine all features
        if all_features:
            features_df = pd.concat(all_features, axis=1)
            logger.info(f"Combined {len(features_df.columns)} total features")
        else:
            features_df = pd.DataFrame(index=df.index)
            logger.warning("No features were extracted")
        
        # Extract labels if specified
        labels = None
        if label_column and label_column in df.columns:
            labels = df[label_column]
            logger.info(f"Extracted labels from {label_column}")
        
        # Save intermediate results if requested
        if save_intermediate:
            self._save_intermediate_results(df, features_df, labels)
        
        return features_df, labels
    
    def _calculate_flesch_reading_ease(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        try:
            sentences = re.split(r'[.!?]+', text)
            words = text.split()
            
            if len(sentences) == 0 or len(words) == 0:
                return 0
            
            # Count syllables (simplified)
            syllable_count = 0
            for word in words:
                word = word.lower()
                syllable_count += max(1, len(re.findall(r'[aeiouy]+', word)))
            
            # Flesch Reading Ease formula
            score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllable_count / len(words))
            return max(0, min(100, score))
        except:
            return 0
    
    def _calculate_smog_index(self, text: str) -> float:
        """Calculate SMOG Index"""
        try:
            sentences = re.split(r'[.!?]+', text)
            if len(sentences) < 3:
                return 0
            
            # Count polysyllabic words (words with 3+ syllables)
            polysyllabic_count = 0
            for word in text.split():
                if len(re.findall(r'[aeiouy]+', word.lower())) >= 3:
                    polysyllabic_count += 1
            
            # SMOG formula
            score = 1.043 * (polysyllabic_count * (30 / len(sentences)))**0.5 + 3.1291
            return score
        except:
            return 0
    
    def _count_positive_words(self, text: str) -> int:
        """Count positive sentiment words"""
        positive_words = self._load_sentiment_words('positive')
        words = str(text).lower().split()
        return sum(1 for word in words if word in positive_words)
    
    def _count_negative_words(self, text: str) -> int:
        """Count negative sentiment words"""
        negative_words = self._load_sentiment_words('negative')
        words = str(text).lower().split()
        return sum(1 for word in words if word in negative_words)
    
    def _load_sentiment_words(self, sentiment_type: str) -> set:
        """Load sentiment words from file"""
        try:
            file_path = Path("data/external/sentiment_lexicons") / f"{sentiment_type}_words.txt"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    words = set(line.strip().lower() for line in f if line.strip())
                return words
        except:
            pass
        
        # Return default word lists if file not found
        if sentiment_type == 'positive':
            return {'good', 'great', 'excellent', 'positive', 'happy', 'wonderful', 
                   'amazing', 'best', 'better', 'love', 'like', 'nice', 'awesome'}
        else:
            return {'bad', 'terrible', 'awful', 'negative', 'sad', 'worst', 
                   'worse', 'hate', 'dislike', 'poor', 'horrible'}
    
    def _load_domain_list(self, filename: str) -> List[str]:
        """Load domain list from file"""
        try:
            file_path = Path("data/external/domain_lists") / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    domains = [line.strip().lower() for line in f if line.strip()]
                return domains
        except:
            pass
        
        return []
    
    def _save_intermediate_results(
        self, 
        df: pd.DataFrame, 
        features_df: pd.DataFrame, 
        labels: Optional[pd.Series]
    ):
        """Save intermediate processing results"""
        import pickle
        
        # Create cache directory
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        processed_data = {
            'original_df': df,
            'features_df': features_df,
            'labels': labels,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        cache_file = cache_dir / f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"Saved intermediate results to {cache_file}")
