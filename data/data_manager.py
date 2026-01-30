import pandas as pd
import numpy as np
import json
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataManager:
    """Centralized data management and versioning"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self._ensure_directory_structure()
        self.config = self._load_config()
        
    def _ensure_directory_structure(self):
        """Create all necessary directories"""
        directories = [
            self.base_path / "raw",
            self.base_path / "raw" / "kaggle_fake_news",
            self.base_path / "raw" / "liar_dataset",
            self.base_path / "raw" / "twitter_fake_news",
            self.base_path / "processed",
            self.base_path / "processed" / "train",
            self.base_path / "processed" / "test",
            self.base_path / "processed" / "validation",
            self.base_path / "processed" / "features",
            self.base_path / "processed" / "statistics",
            self.base_path / "external",
            self.base_path / "external" / "sentiment_lexicons",
            self.base_path / "external" / "domain_lists",
            self.base_path / "external" / "embeddings",
            self.base_path / "cache",
            self.base_path / "cache" / "temporary_features",
            self.base_path / "cache" / "processed_chunks"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Load data configuration"""
        config_path = self.base_path / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default data configuration"""
        return {
            "data_sources": {
                "primary": "fake_news.csv",
                "external": {
                    "kaggle": "kaggle_fake_news/train.csv",
                    "liar": "liar_dataset/train.tsv",
                    "twitter": "twitter_fake_news/tweets.csv"
                }
            },
            "columns": {
                "text": ["text", "content", "article", "statement"],
                "title": ["title", "headline"],
                "label": ["label", "target", "is_fake", "classification"],
                "source": ["source", "publisher", "domain"],
                "date": ["date", "timestamp", "published_date"],
                "author": ["author", "speaker", "writer"]
            },
            "preprocessing": {
                "text_cleaning": {
                    "remove_urls": True,
                    "remove_mentions": True,
                    "remove_hashtags": True,
                    "remove_numbers": False,
                    "remove_special_chars": True,
                    "lowercase": True,
                    "remove_stopwords": True,
                    "stem": True,
                    "lemmatize": False
                },
                "feature_engineering": {
                    "extract_text_features": True,
                    "extract_metadata_features": True,
                    "use_external_features": True
                }
            },
            "splitting": {
                "test_size": 0.2,
                "validation_size": 0.1,
                "random_state": 42,
                "stratify": True
            }
        }
    
    def load_raw_data(
        self, 
        source: str = "primary",
        sample_size: Optional[int] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Load raw data from specified source
        
        Args:
            source: Data source (primary, kaggle, liar, twitter)
            sample_size: Number of samples to load
            random_state: Random seed for sampling
        
        Returns:
            Loaded DataFrame
        """
        if source == "primary":
            file_path = self.base_path / "raw" / self.config["data_sources"]["primary"]
        else:
            external_sources = self.config["data_sources"]["external"]
            if source in external_sources:
                file_path = self.base_path / "raw" / external_sources[source]
            else:
                raise ValueError(f"Unknown data source: {source}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        # Load based on file extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.tsv':
            df = pd.read_csv(file_path, sep='\t')
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path, orient='records')
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Sample if requested
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(sample_size, random_state=random_state)
            logger.info(f"Sampled {sample_size} records")
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def load_processed_data(
        self,
        dataset_type: str = "train",
        load_features: bool = False,
        load_labels: bool = True
    ) -> Union[pd.DataFrame, Tuple]:
        """
        Load processed data
        
        Args:
            dataset_type: Type of dataset (train, test, validation)
            load_features: Whether to load numpy features
            load_labels: Whether to load labels
        
        Returns:
            DataFrame or tuple of data components
        """
        data_dir = self.base_path / "processed" / dataset_type
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
        
        # Load CSV data
        csv_path = data_dir / f"{dataset_type}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {dataset_type} data: {len(df)} records")
        else:
            df = None
        
        # Load numpy features if requested
        features = None
        if load_features:
            features_path = data_dir / f"{dataset_type}_features.npy"
            if features_path.exists():
                features = np.load(features_path)
                logger.info(f"Loaded features: {features.shape}")
        
        # Load numpy labels if requested
        labels = None
        if load_labels:
            labels_path = data_dir / f"{dataset_type}_labels.npy"
            if labels_path.exists():
                labels = np.load(labels_path)
                logger.info(f"Loaded labels: {labels.shape}")
        
        # Load metadata
        metadata_path = data_dir / f"{dataset_type}_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        if load_features or load_labels:
            return df, features, labels, metadata
        else:
            return df, metadata
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Save processed data
        
        Args:
            df: DataFrame to save
            dataset_type: Type of dataset (train, test, validation)
            features: Optional numpy features
            labels: Optional numpy labels
            metadata: Optional metadata
        """
        data_dir = self.base_path / "processed" / dataset_type
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = data_dir / f"{dataset_type}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {dataset_type} data to {csv_path}")
        
        # Save numpy features
        if features is not None:
            features_path = data_dir / f"{dataset_type}_features.npy"
            np.save(features_path, features)
            logger.info(f"Saved features to {features_path}")
        
        # Save numpy labels
        if labels is not None:
            labels_path = data_dir / f"{dataset_type}_labels.npy"
            np.save(labels_path, labels)
            logger.info(f"Saved labels to {labels_path}")
        
        # Save metadata
        if metadata is not None:
            metadata_path = data_dir / f"{dataset_type}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
    
    def save_features(
        self,
        feature_name: str,
        features: Any,
        metadata: Optional[Dict] = None
    ):
        """
        Save feature objects (vectorizer, scaler, etc.)
        
        Args:
            feature_name: Name of feature object
            features: Feature object to save
            metadata: Optional metadata
        """
        features_dir = self.base_path / "processed" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature object
        feature_path = features_dir / f"{feature_name}.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(features, f)
        logger.info(f"Saved {feature_name} to {feature_path}")
        
        # Save metadata if provided
        if metadata is not None:
            metadata_path = features_dir / f"{feature_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_features(self, feature_name: str) -> Any:
        """
        Load feature objects
        
        Args:
            feature_name: Name of feature object
        
        Returns:
            Loaded feature object
        """
        feature_path = self.base_path / "processed" / "features" / f"{feature_name}.pkl"
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        
        with open(feature_path, 'rb') as f:
            features = pickle.load(f)
        
        logger.info(f"Loaded {feature_name} from {feature_path}")
        return features
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data statistics
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "basic": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "generated_at": datetime.now().isoformat()
            },
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": {
                "count": df.isnull().sum().to_dict(),
                "percentage": (df.isnull().sum() / len(df) * 100).to_dict()
            },
            "numerical_summary": {},
            "categorical_summary": {}
        }
        
        # Numerical columns statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            stats["numerical_summary"] = {
                "columns": numerical_cols.tolist(),
                "count": len(numerical_cols),
                "descriptive_stats": df[numerical_cols].describe().to_dict()
            }
            
            # Correlation matrix (for numerical columns)
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                stats["numerical_summary"]["correlation_matrix"] = corr_matrix.to_dict()
        
        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            categorical_stats = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                categorical_stats[col] = {
                    "unique_values": df[col].nunique(),
                    "most_common": {
                        "value": value_counts.index[0] if len(value_counts) > 0 else None,
                        "count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                        "percentage": (value_counts.iloc[0] / len(df) * 100) if len(value_counts) > 0 else 0
                    },
                    "top_10_values": value_counts.head(10).to_dict()
                }
            stats["categorical_summary"] = categorical_stats
        
        # Text columns analysis (if any)
        text_cols = self._identify_text_columns(df)
        if text_cols:
            text_stats = {}
            for col in text_cols:
                text_stats[col] = {
                    "average_length": df[col].str.len().mean(),
                    "min_length": df[col].str.len().min(),
                    "max_length": df[col].str.len().max(),
                    "empty_strings": (df[col] == "").sum(),
                    "whitespace_only": (df[col].str.strip() == "").sum()
                }
            stats["text_summary"] = text_stats
        
        return stats
    
    def save_data_statistics(self, df: pd.DataFrame, name: str = "data_statistics"):
        """
        Save data statistics to file
        
        Args:
            df: Input DataFrame
            name: Name for statistics file
        """
        stats = self.get_data_statistics(df)
        
        stats_dir = self.base_path / "processed" / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = stats_dir / f"{name}.json"
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Save summary as markdown
        md_path = stats_dir / f"{name}.md"
        self._save_statistics_markdown(stats, md_path)
        
        logger.info(f"Saved data statistics to {json_path}")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names based on configuration"""
        column_mapping = self.config.get("columns", {})
        
        # Create mapping from possible names to standardized names
        std_mapping = {}
        for std_name, possible_names in column_mapping.items():
            if isinstance(possible_names, str):
                std_mapping[possible_names.lower()] = std_name
            elif isinstance(possible_names, list):
                for name in possible_names:
                    std_mapping[name.lower()] = std_name
        
        # Rename columns
        new_columns = []
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in std_mapping:
                new_columns.append(std_mapping[col_lower])
            else:
                new_columns.append(col)
        
        df.columns = new_columns
        return df
    
    def _identify_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify text columns in DataFrame"""
        text_cols = []
        
        # Check columns that might contain text
        for col in df.columns:
            # Check if column contains strings and has reasonable length
            if df[col].dtype == 'object':
                # Check if values look like text (not just categorical codes)
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    # Check average length and character diversity
                    avg_length = sample.astype(str).str.len().mean()
                    if avg_length > 10:  # Reasonable threshold for text
                        text_cols.append(col)
        
        return text_cols
    
    def _save_statistics_markdown(self, stats: Dict, output_path: Path):
        """Save statistics as markdown report"""
        md_content = f"""# Data Statistics Report
Generated: {stats['basic']['generated_at']}

## Basic Information
- **Total Rows**: {stats['basic']['rows']:,}
- **Total Columns**: {stats['basic']['columns']}
- **Memory Usage**: {stats['basic']['memory_usage_mb']:.2f} MB

## Missing Values
| Column | Missing Count | Missing Percentage |
|--------|---------------|-------------------|
"""
        
        # Add missing values table
        for col, missing_pct in stats['missing_values']['percentage'].items():
            missing_count = stats['missing_values']['count'][col]
            if missing_count > 0:
                md_content += f"| {col} | {missing_count:,} | {missing_pct:.1f}% |\n"
        
        # Add numerical summary
        if stats['numerical_summary']:
            md_content += f"\n## Numerical Columns ({stats['numerical_summary']['count']})\n\n"
            for col in stats['numerical_summary']['columns']:
                col_stats = stats['numerical_summary']['descriptive_stats'][col]
                md_content += f"### {col}\n"
                md_content += f"- Mean: {col_stats['mean']:.2f}\n"
                md_content += f"- Std: {col_stats['std']:.2f}\n"
                md_content += f"- Min: {col_stats['min']:.2f}\n"
                md_content += f"- 25%: {col_stats['25%']:.2f}\n"
                md_content += f"- 50%: {col_stats['50%']:.2f}\n"
                md_content += f"- 75%: {col_stats['75%']:.2f}\n"
                md_content += f"- Max: {col_stats['max']:.2f}\n\n"
        
        # Add categorical summary
        if stats['categorical_summary']:
            md_content += f"\n## Categorical Columns\n\n"
            for col, col_stats in stats['categorical_summary'].items():
                md_content += f"### {col}\n"
                md_content += f"- Unique Values: {col_stats['unique_values']}\n"
                md_content += f"- Most Common: '{col_stats['most_common']['value']}' "
                md_content += f"({col_stats['most_common']['count']:,} records, "
                md_content += f"{col_stats['most_common']['percentage']:.1f}%)\n\n"
        
        with open(output_path, 'w') as f:
            f.write(md_content)
    
    def get_data_hash(self, df: pd.DataFrame) -> str:
        """
        Calculate hash of data for versioning
        
        Args:
            df: Input DataFrame
        
        Returns:
            Data hash string
        """
        # Convert DataFrame to bytes and calculate hash
        data_bytes = df.to_csv(index=False).encode('utf-8')
        return hashlib.sha256(data_bytes).hexdigest()[:16]
    
    def backup_data(self, source_dir: str, backup_name: str):
        """
        Backup data directory
        
        Args:
            source_dir: Source directory to backup
            backup_name: Name for backup
        """
        import shutil
        from datetime import datetime
        
        source_path = self.base_path / source_dir
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_path}")
        
        # Create backup directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.base_path / "backups" / f"{backup_name}_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy data
        shutil.copytree(source_path, backup_dir / source_dir)
        logger.info(f"Backed up {source_dir} to {backup_dir}")
