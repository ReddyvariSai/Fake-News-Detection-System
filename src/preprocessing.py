import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Main data preprocessing class for fake news detection"""
    
    def __init__(self, language='english', remove_stopwords=True, 
                 lemmatize=True, min_word_length=2):
        """
        Initialize the preprocessor
        
        Parameters:
        -----------
        language : str
            Language for text processing
        remove_stopwords : bool
            Whether to remove stopwords
        lemmatize : bool
            Whether to lemmatize words
        min_word_length : int
            Minimum word length to keep
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        
        # Initialize NLP tools
        self._download_nltk_resources()
        self._initialize_nlp_tools()
        
    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        required_resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            except LookupError:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource)
    
    def _initialize_nlp_tools(self):
        """Initialize all NLP tools"""
        self.stop_words = set(stopwords.words(self.language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Add custom stopwords for news context
        custom_stopwords = {
            'said', 'say', 'says', 'according', 'would', 'could', 'also',
            'like', 'get', 'go', 'know', 'think', 'make', 'take', 'see',
            'come', 'want', 'look', 'use', 'find', 'give', 'tell', 'new',
            'good', 'well', 'first', 'last', 'one', 'two', 'three', 'four'
        }
        self.stop_words.update(custom_stopwords)
    
    def clean_text(self, text, remove_urls=True, remove_emails=True, 
                   remove_numbers=True, remove_special_chars=True):
        """
        Clean a single text string
        
        Parameters:
        -----------
        text : str
            Raw text input
        remove_urls : bool
            Whether to remove URLs
        remove_emails : bool
            Whether to remove email addresses
        remove_numbers : bool
            Whether to remove numbers
        remove_special_chars : bool
            Whether to remove special characters
            
        Returns:
        --------
        str : Cleaned text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        if remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters
        if remove_special_chars:
            # Keep basic punctuation
            text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        
        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text, remove_stopwords=None, lemmatize=None):
        """
        Tokenize and process text
        
        Parameters:
        -----------
        text : str
            Cleaned text
        remove_stopwords : bool, optional
            Override class setting
        lemmatize : bool, optional
            Override class setting
            
        Returns:
        --------
        list : Processed tokens
        """
        if remove_stopwords is None:
            remove_stopwords = self.remove_stopwords
        if lemmatize is None:
            lemmatize = self.lemmatize
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Process tokens
        processed_tokens = []
        for token in tokens:
            # Remove very short tokens
            if len(token) < self.min_word_length:
                continue
            
            # Remove stopwords if enabled
            if remove_stopwords and token in self.stop_words:
                continue
            
            # Apply lemmatization or stemming
            if lemmatize:
                token = self.lemmatizer.lemmatize(token)
            else:
                token = self.stemmer.stem(token)
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    def preprocess_pipeline(self, text, return_tokens=False):
        """
        Complete preprocessing pipeline for a single text
        
        Parameters:
        -----------
        text : str
            Raw text input
        return_tokens : bool
            Whether to return tokens or joined text
            
        Returns:
        --------
        str or list : Processed text or tokens
        """
        # Step 1: Clean text
        cleaned_text = self.clean_text(text)
        
        # Step 2: Tokenize and process
        tokens = self.tokenize_text(cleaned_text)
        
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='text', label_column='label', 
                            save_processed=False, output_path=None):
        """
        Preprocess entire dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        text_column : str
            Name of text column
        label_column : str
            Name of label column
        save_processed : bool
            Whether to save processed data
        output_path : str, optional
            Path to save processed data
            
        Returns:
        --------
        pandas.DataFrame : Processed dataframe
        """
        print("="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Make a copy
        df_processed = df.copy()
        
        # Check for required columns
        if text_column not in df_processed.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataframe")
        
        print(f"Processing {len(df_processed)} samples...")
        
        # Handle missing values
        initial_count = len(df_processed)
        df_processed = df_processed.dropna(subset=[text_column])
        removed_count = initial_count - len(df_processed)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with missing text")
        
        # Apply preprocessing pipeline
        print("Cleaning and processing text...")
        df_processed['cleaned_text'] = df_processed[text_column].apply(
            lambda x: self.preprocess_pipeline(x, return_tokens=False)
        )
        
        # Create tokens column
        df_processed['tokens'] = df_processed[text_column].apply(
            lambda x: self.preprocess_pipeline(x, return_tokens=True)
        )
        
        # Calculate text statistics
        print("Calculating text statistics...")
        df_processed['text_length'] = df_processed['cleaned_text'].apply(len)
        df_processed['word_count'] = df_processed['tokens'].apply(len)
        
        # Remove very short texts
        min_words = 10
        df_processed = df_processed[df_processed['word_count'] >= min_words]
        print(f"Removed texts with less than {min_words} words")
        
        print(f"\nPreprocessing completed!")
        print(f"Original samples: {initial_count}")
        print(f"Processed samples: {len(df_processed)}")
        
        # Save processed data if requested
        if save_processed:
            if output_path is None:
                output_path = 'data/processed/cleaned_news.csv'
            df_processed.to_csv(output_path, index=False)
            print(f"Processed data saved to: {output_path}")
        
        return df_processed
    
    def split_dataset(self, df, text_column='cleaned_text', label_column='label',
                     test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Processed dataframe
        text_column : str
            Name of text column
        label_column : str
            Name of label column
        test_size : float
            Proportion of test data
        val_size : float
            Proportion of validation data from training data
        random_state : int
            Random seed
            
        Returns:
        --------
        tuple : (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataframe")
        
        X = df[text_column]
        y = df[label_column]
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and validation
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, 
            random_state=random_state, stratify=y_temp
        )
        
        print("\nDataset Split Summary:")
        print("-" * 40)
        print(f"Training samples:   {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
        print(f"Validation samples: {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
        print(f"Test samples:       {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
        
        if label_column in df.columns:
            print(f"\nLabel distribution in splits:")
            splits = [
                ('Training', y_train),
                ('Validation', y_val),
                ('Test', y_test)
            ]
            
            for name, split_labels in splits:
                if len(split_labels) > 0:
                    real_pct = (split_labels == 0).sum() / len(split_labels) * 100
                    fake_pct = (split_labels == 1).sum() / len(split_labels) * 100
                    print(f"{name:12} - Real: {real_pct:.1f}%, Fake: {fake_pct:.1f}%")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

# Advanced preprocessor with more features
class EnhancedDataPreprocessor(DataPreprocessor):
    """Enhanced preprocessor with additional features for fake news detection"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Fake news specific dictionaries
        self.sensational_words = {
            'shocking', 'breaking', 'secret', 'amazing', 'unbelievable',
            'hidden', 'cover-up', 'exposed', 'scandal', 'leaked', 'urgent',
            'emergency', 'warning', 'alert', 'dangerous', 'explosive',
            'miracle', 'instant', 'guaranteed', 'classified', 'forbidden',
            'banned', 'controversial', 'sensational', 'bombshell'
        }
        
        self.conspiracy_terms = {
            'conspiracy', 'deep state', 'new world order', 'illuminati',
            'false flag', 'hoax', 'mainstream media', 'establishment',
            'elite', 'globalists', 'shadow government', 'cover-up'
        }
        
        self.credibility_indicators = {
            'according to study', 'research shows', 'scientists found',
            'peer-reviewed', 'published in', 'data indicates', 'statistics show',
            'expert said', 'official report', 'clinical trial', 'meta-analysis'
        }
    
    def extract_text_features(self, text):
        """
        Extract advanced text features for fake news detection
        
        Parameters:
        -----------
        text : str
            Cleaned text
            
        Returns:
        --------
        dict : Dictionary of advanced features
        """
        features = {}
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        
        # Word statistics
        words = text.split()
        if words:
            word_lengths = [len(word) for word in words]
            features['avg_word_length'] = np.mean(word_lengths)
            features['max_word_length'] = max(word_lengths)
            features['unique_words'] = len(set(words))
            features['lexical_diversity'] = features['unique_words'] / max(len(words), 1)
        else:
            features['avg_word_length'] = 0
            features['max_word_length'] = 0
            features['unique_words'] = 0
            features['lexical_diversity'] = 0
        
        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['quote_count'] = text.count('"') + text.count("'")
        
        # Fake news specific features
        text_lower = text.lower()
        
        # Sensationalism score
        features['sensational_score'] = sum(
            1 for word in self.sensational_words if word in text_lower
        )
        
        # Conspiracy score
        features['conspiracy_score'] = sum(
            1 for term in self.conspiracy_terms if term in text_lower
        )
        
        # Credibility score
        features['credibility_score'] = sum(
            1 for term in self.credibility_indicators if term in text_lower
        )
        
        # Emotional language detection
        emotional_words = ['outrageous', 'horrifying', 'terrifying', 'disgusting',
                          'frightening', 'appalling', 'shameful', 'scandalous']
        features['emotional_score'] = sum(
            1 for word in emotional_words if word in text_lower
        )
        
        # Modal verbs (common in speculative/fake news)
        modal_verbs = ['might', 'could', 'would', 'may', 'can', 'should', 'must']
        features['modal_verb_count'] = sum(
            text_lower.count(verb) for verb in modal_verbs
        )
        
        # First person pronouns (common in opinion pieces)
        first_person = ['i', 'we', 'our', 'us', 'my', 'mine', 'ours']
        features['first_person_count'] = sum(
            text_lower.count(pronoun) for pronoun in first_person
        )
        
        # Sentiment analysis
        try:
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment_polarity'] = 0
            features['sentiment_subjectivity'] = 0
        
        # Readability features
        if features['word_count'] > 0 and features['sentence_count'] > 0:
            features['avg_words_per_sentence'] = features['word_count'] / features['sentence_count']
        else:
            features['avg_words_per_sentence'] = 0
        
        return features
    
    def extract_all_features(self, df, text_column='cleaned_text'):
        """
        Extract all advanced features for dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Processed dataframe
        text_column : str
            Name of text column
            
        Returns:
        --------
        pandas.DataFrame : Dataframe with extracted features
        """
        print("Extracting advanced text features...")
        
        # Extract features for each text
        features_list = []
        for text in df[text_column]:
            features = self.extract_text_features(text)
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Combine with original dataframe
        result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        # Calculate composite scores
        result_df['fake_news_score'] = (
            result_df['sensational_score'] * 0.3 +
            result_df['conspiracy_score'] * 0.3 +
            result_df['emotional_score'] * 0.2 +
            (result_df['exclamation_count'] / result_df['word_count'].clip(lower=1)) * 0.2
        )
        
        result_df['credibility_score_norm'] = (
            result_df['credibility_score'] / result_df['word_count'].clip(lower=1)
        )
        
        print(f"Extracted {len(features_df.columns)} advanced features")
        
        return result_df

# Usage example
if __name__ == "__main__":
    # Example usage
    preprocessor = EnhancedDataPreprocessor()
    
    # Sample data
    sample_texts = [
        "BREAKING: Shocking discovery cures cancer overnight! Doctors hate this!",
        "According to a new study published in Nature, regular exercise reduces health risks."
    ]
    
    print("Sample preprocessing:")
    for text in sample_texts:
        print(f"\nOriginal: {text}")
        cleaned = preprocessor.preprocess_pipeline(text)
        print(f"Cleaned: {cleaned}")
        
        features = preprocessor.extract_text_features(text)
        print(f"Features: {features}")
