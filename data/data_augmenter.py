import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import json
import random
from textaugment import EDA
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from googletrans import Translator
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataAugmenter:
    """Data augmentation for text classification tasks"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.translator = None
        self.paraphrase_pipeline = None
        self._setup_augmenters()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load augmentation configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default augmentation configuration"""
        return {
            "augmentation_strategies": {
                "synonym_replacement": {
                    "enabled": True,
                    "aug_p": 0.3,
                    "max_aug": 5
                },
                "random_insertion": {
                    "enabled": True,
                    "aug_p": 0.3,
                    "max_aug": 5
                },
                "random_swap": {
                    "enabled": True,
                    "aug_p": 0.3,
                    "max_aug": 5
                },
                "random_deletion": {
                    "enabled": True,
                    "aug_p": 0.3,
                    "max_aug": 5
                },
                "back_translation": {
                    "enabled": False,
                    "intermediate_languages": ["fr", "de", "es"]
                },
                "paraphrasing": {
                    "enabled": False,
                    "model_name": "t5-base"
                },
                "spelling_error": {
                    "enabled": False,
                    "aug_char_p": 0.3,
                    "aug_word_p": 0.3
                }
            },
            "class_specific_augmentation": {
                "enabled": True,
                "augment_minority_only": True,
                "target_samples_per_class": None,
                "max_augmentation_factor": 2.0
            },
            "quality_control": {
                "validate_augmented_text": True,
                "min_text_length": 10,
                "max_text_length": 1000,
                "remove_duplicates": True
            },
            "output": {
                "save_augmented_data": True,
                "output_dir": "data/processed/augmented/",
                "format": "csv"
            }
        }
    
    def _setup_augmenters(self):
        """Setup augmentation models and tools"""
        self.augmenters = {}
        
        # EDA augmenter (fast, rule-based)
        if self.config["augmentation_strategies"]["synonym_replacement"]["enabled"]:
            try:
                self.augmenters["synonym"] = EDA()
            except:
                logger.warning("Failed to initialize EDA augmenter")
        
        # nlpaug augmenters
        try:
            if self.config["augmentation_strategies"]["synonym_replacement"]["enabled"]:
                self.augmenters["nlpaug_synonym"] = naw.SynonymAug(
                    aug_src='wordnet',
                    aug_p=self.config["augmentation_strategies"]["synonym_replacement"]["aug_p"]
                )
            
            if self.config["augmentation_strategies"]["random_insertion"]["enabled"]:
                self.augmenters["random_insert"] = naw.RandomWordAug(
                    action="insert",
                    aug_p=self.config["augmentation_strategies"]["random_insertion"]["aug_p"]
                )
            
            if self.config["augmentation_strategies"]["random_swap"]["enabled"]:
                self.augmenters["random_swap"] = naw.RandomWordAug(
                    action="swap",
                    aug_p=self.config["augmentation_strategies"]["random_swap"]["aug_p"]
                )
            
            if self.config["augmentation_strategies"]["random_deletion"]["enabled"]:
                self.augmenters["random_delete"] = naw.RandomWordAug(
                    action="delete",
                    aug_p=self.config["augmentation_strategies"]["random_deletion"]["aug_p"]
                )
            
            if self.config["augmentation_strategies"]["spelling_error"]["enabled"]:
                self.augmenters["spelling"] = nac.RandomCharAug(
                    aug_char_p=self.config["augmentation_strategies"]["spelling_error"]["aug_char_p"],
                    aug_word_p=self.config["augmentation_strategies"]["spelling_error"]["aug_word_p"]
                )
        
        except ImportError:
            logger.warning("nlpaug not installed, skipping advanced augmentations")
        
        # Back translation setup
        if self.config["augmentation_strategies"]["back_translation"]["enabled"]:
            try:
                self.translator = Translator()
            except:
                logger.warning("Failed to initialize translator for back translation")
        
        # Paraphrasing setup
        if self.config["augmentation_strategies"]["paraphrasing"]["enabled"]:
            try:
                model_name = self.config["augmentation_strategies"]["paraphrasing"]["model_name"]
                self.paraphrase_pipeline = pipeline(
                    "text2text-generation",
                    model=f"Vamsi/{model_name}-paraphrase"
                )
            except:
                logger.warning("Failed to initialize paraphrase model")
    
    def augment_dataset(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        strategies: List[str] = None,
        n_augmentations: int = 1,
        balance_classes: bool = True
    ) -> pd.DataFrame:
        """
        Augment dataset using specified strategies
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column
            strategies: List of augmentation strategies to use
            n_augmentations: Number of augmented versions per sample
            balance_classes: Whether to balance classes through augmentation
        
        Returns:
            Augmented DataFrame
        """
        logger.info(f"Starting augmentation for {len(df)} samples")
        
        if strategies is None:
            strategies = self._get_available_strategies()
        
        # Validate input
        self._validate_augmentation_input(df, text_column, label_column)
        
        # Apply class-specific augmentation if enabled
        if balance_classes and self.config["class_specific_augmentation"]["enabled"]:
            df_augmented = self._balance_classes_through_augmentation(
                df, text_column, label_column, strategies, n_augmentations
            )
        else:
            # Augment all samples
            augmented_samples = []
            
            for _, row in df.iterrows():
                text = row[text_column]
                label = row[label_column]
                
                # Generate augmented versions
                for _ in range(n_augmentations):
                    augmented_text = self._augment_text(text, strategies)
                    
                    if augmented_text and self._validate_augmented_text(augmented_text):
                        new_row = row.copy()
                        new_row[text_column] = augmented_text
                        new_row['is_augmented'] = True
                        new_row['augmentation_source'] = 'original'
                        augmented_samples.append(new_row)
            
            # Combine original and augmented
            df_original = df.copy()
            df_original['is_augmented'] = False
            df_original['augmentation_source'] = 'original'
            
            df_augmented = pd.concat([df_original, pd.DataFrame(augmented_samples)], ignore_index=True)
        
        # Quality control
        df_augmented = self._apply_quality_control(df_augmented, text_column)
        
        # Save if configured
        if self.config["output"]["save_augmented_data"]:
            self._save_augmented_data(df_augmented, strategies)
        
        logger.info(f"Augmentation complete: {len(df_augmented)} total samples "
                   f"({len(df_augmented) - len(df)} augmented)")
        
        return df_augmented
    
    def _balance_classes_through_augmentation(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        strategies: List[str],
        n_augmentations: int
    ) -> pd.DataFrame:
        """Balance classes by augmenting minority classes"""
        class_counts = df[label_column].value_counts()
        max_count = class_counts.max()
        
        augmented_samples = []
        df_original = df.copy()
        df_original['is_augmented'] = False
        df_original['augmentation_source'] = 'original'
        
        augment_minority_only = self.config["class_specific_augmentation"]["augment_minority_only"]
        target_samples = self.config["class_specific_augmentation"]["target_samples_per_class"]
        max_factor = self.config["class_specific_augmentation"]["max_augmentation_factor"]
        
        for class_label, count in class_counts.items():
            class_df = df[df[label_column] == class_label]
            
            # Determine how many augmented samples to create
            if target_samples:
                needed_samples = max(0, target_samples - count)
            elif augment_minority_only and count < max_count:
                needed_samples = max_count - count
            else:
                needed_samples = int(count * (n_augmentations - 1))
            
            # Limit augmentation factor
            max_augmented = int(count * max_factor)
            needed_samples = min(needed_samples, max_augmented)
            
            if needed_samples > 0:
                logger.info(f"Augmenting class {class_label}: {count} -> {count + needed_samples}")
                
                # Create augmented samples
                created_samples = 0
                while created_samples < needed_samples:
                    # Sample from original class data
                    sample = class_df.sample(1).iloc[0]
                    text = sample[text_column]
                    
                    # Generate augmented version
                    augmented_text = self._augment_text(text, strategies)
                    
                    if augmented_text and self._validate_augmented_text(augmented_text):
                        new_row = sample.copy()
                        new_row[text_column] = augmented_text
                        new_row['is_augmented'] = True
                        new_row['augmentation_source'] = f'class_{class_label}_balance'
                        augmented_samples.append(new_row)
                        created_samples += 1
        
        # Combine all samples
        df_augmented = pd.concat([df_original, pd.DataFrame(augmented_samples)], ignore_index=True)
        return df_augmented
    
    def _augment_text(self, text: str, strategies: List[str]) -> str:
        """Apply augmentation strategies to text"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return text
        
        augmented_text = text
        
        for strategy in strategies:
            try:
                if strategy == "synonym_replacement" and "nlpaug_synonym" in self.augmenters:
                    augmented_text = self.augmenters["nlpaug_synonym"].augment(augmented_text)
                
                elif strategy == "random_insertion" and "random_insert" in self.augmenters:
                    augmented_text = self.augmenters["random_insert"].augment(augmented_text)
                
                elif strategy == "random_swap" and "random_swap" in self.augmenters:
                    augmented_text = self.augmenters["random_swap"].augment(augmented_text)
                
                elif strategy == "random_deletion" and "random_delete" in self.augmenters:
                    augmented_text = self.augmenters["random_delete"].augment(augmented_text)
                
                elif strategy == "back_translation" and self.translator:
                    augmented_text = self._back_translate(augmented_text)
                
                elif strategy == "paraphrasing" and self.paraphrase_pipeline:
                    augmented_text = self._paraphrase(augmented_text)
                
                elif strategy == "spelling_error" and "spelling" in self.augmenters:
                    augmented_text = self.augmenters["spelling"].augment(augmented_text)
            
            except Exception as e:
                logger.warning(f"Error in {strategy} augmentation: {e}")
                continue
        
        return augmented_text
    
    def _back_translate(self, text: str) -> str:
        """Back translation augmentation"""
        try:
            # Translate to intermediate language
            intermediate_lang = random.choice(
                self.config["augmentation_strategies"]["back_translation"]["intermediate_languages"]
            )
            
            translated = self.translator.translate(text, dest=intermediate_lang).text
            
            # Translate back to English
            back_translated = self.translator.translate(translated, dest='en').text
            
            return back_translated
        
        except Exception as e:
            logger.warning(f"Back translation failed: {e}")
            return text
    
    def _paraphrase(self, text: str) -> str:
        """Paraphrase text using T5 model"""
        try:
            input_text = f"paraphrase: {text}"
            result = self.paraphrase_pipeline(input_text, max_length=100, num_return_sequences=1)
            
            if result and len(result) > 0:
                return result[0]['generated_text']
        
        except Exception as e:
            logger.warning(f"Paraphrasing failed: {e}")
        
        return text
    
    def _validate_augmentation_input(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str
    ):
        """Validate input for augmentation"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found")
        
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found")
        
        # Check text quality
        empty_texts = df[text_column].isna() | (df[text_column].str.strip() == "")
        if empty_texts.any():
            logger.warning(f"Found {empty_texts.sum()} empty texts")
    
    def _validate_augmented_text(self, text: str) -> bool:
        """Validate augmented text meets quality criteria"""
        if not isinstance(text, str):
            return False
        
        text = text.strip()
        
        # Check length
        min_len = self.config["quality_control"]["min_text_length"]
        max_len = self.config["quality_control"]["max_text_length"]
        
        if len(text) < min_len or len(text) > max_len:
            return False
        
        # Check for obvious issues
        if text.lower() in ["nan", "null", "none", ""]:
            return False
        
        return True
    
    def _apply_quality_control(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Apply quality control to augmented data"""
        # Remove invalid texts
        valid_mask = df[text_column].apply(self._validate_augmented_text)
        df = df[valid_mask].copy()
        
        # Remove duplicates if configured
        if self.config["quality_control"]["remove_duplicates"]:
            # Keep first occurrence of duplicate texts
            df = df.drop_duplicates(subset=[text_column], keep='first')
        
        return df
    
    def _get_available_strategies(self) -> List[str]:
        """Get list of available augmentation strategies"""
        strategies = []
        config_strategies = self.config["augmentation_strategies"]
        
        for strategy, params in config_strategies.items():
            if params.get("enabled", False):
                strategies.append(strategy)
        
        return strategies
    
    def _save_augmented_data(self, df: pd.DataFrame, strategies: List[str]):
        """Save augmented data to disk"""
        output_dir = Path(self.config["output"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        strategy_str = "_".join(strategies)[:50]  # Limit length
        
        # Save metadata
        metadata = {
            "created_at": pd.Timestamp.now().isoformat(),
            "strategies": strategies,
            "original_samples": len(df[df['is_augmented'] == False]),
            "augmented_samples": len(df[df['is_augmented'] == True]),
            "total_samples": len(df)
        }
        
        metadata_file = output_dir / f"augmentation_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save data
        format_type = self.config["output"]["format"]
        if format_type == "csv":
            data_file = output_dir / f"augmented_data_{strategy_str}_{timestamp}.csv"
            df.to_csv(data_file, index=False)
        elif format_type == "parquet":
            data_file = output_dir / f"augmented_data_{strategy_str}_{timestamp}.parquet"
            df.to_parquet(data_file, index=False)
        
        logger.info(f"Saved augmented data to {data_file}")
        logger.info(f"Saved metadata to {metadata_file}")
    
    def create_augmentation_pipeline(
        self,
        input_path: str,
        output_path: str,
        text_column: str = "text",
        label_column: str = "label",
        strategies: List[str] = None,
        balance_classes: bool = True
    ):
        """
        Complete augmentation pipeline from file to file
        
        Args:
            input_path: Path to input data file
            output_path: Path to save augmented data
            text_column: Name of text column
            label_column: Name of label column
            strategies: Augmentation strategies to use
            balance_classes: Whether to balance classes
        """
        # Load data
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Augment data
        df_augmented = self.augment_dataset(
            df=df,
            text_column=text_column,
            label_column=label_column,
            strategies=strategies,
            balance_classes=balance_classes
        )
        
        # Save augmented data
        logger.info(f"Saving augmented data to {output_path}")
        df_augmented.to_csv(output_path, index=False)
        
        return df_augmented
