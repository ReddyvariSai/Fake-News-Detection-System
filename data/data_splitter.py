import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import json
import yaml
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataSplitter:
    """Advanced data splitting strategies for fake news detection"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.split_history = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load splitting configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default splitting configuration"""
        return {
            "splitting_strategies": {
                "simple_split": {
                    "test_size": 0.2,
                    "validation_size": 0.1,
                    "random_state": 42,
                    "stratify": True,
                    "shuffle": True
                },
                "time_based_split": {
                    "date_column": "date",
                    "test_time_period": "30D",  # 30 days for test
                    "validation_time_period": "15D",  # 15 days for validation
                    "chronological": True
                },
                "cross_validation": {
                    "n_splits": 5,
                    "random_state": 42,
                    "shuffle": True,
                    "stratify": True
                },
                "domain_based_split": {
                    "domain_column": "source",
                    "test_domain_ratio": 0.2,
                    "holdout_domains": []  # Specific domains to hold out
                },
                "author_based_split": {
                    "author_column": "author",
                    "test_author_ratio": 0.2,
                    "holdout_authors": []  # Specific authors to hold out
                }
            },
            "data_requirements": {
                "min_samples_per_split": 100,
                "max_class_imbalance": 0.3,
                "ensure_all_classes": True
            },
            "output_format": {
                "save_splits": True,
                "save_indices": True,
                "save_metadata": True,
                "output_dir": "data/processed/splits/",
                "format": "csv"  # csv, parquet, or pickle
            }
        }
    
    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        strategy: str = "simple_split",
        text_column: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Split data using specified strategy
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            strategy: Splitting strategy (simple_split, time_based, cross_validation, domain_based, author_based)
            text_column: Name of text column (for some strategies)
            **kwargs: Additional strategy-specific parameters
        
        Returns:
            Dictionary with splits and metadata
        """
        logger.info(f"Splitting data using {strategy} strategy")
        
        # Validate input data
        self._validate_split_input(df, target_column, strategy, **kwargs)
        
        # Apply selected strategy
        if strategy == "simple_split":
            splits = self._simple_split(df, target_column, **kwargs)
        elif strategy == "time_based_split":
            splits = self._time_based_split(df, target_column, **kwargs)
        elif strategy == "cross_validation":
            splits = self._cross_validation_split(df, target_column, **kwargs)
        elif strategy == "domain_based_split":
            splits = self._domain_based_split(df, target_column, **kwargs)
        elif strategy == "author_based_split":
            splits = self._author_based_split(df, target_column, **kwargs)
        elif strategy == "hybrid_split":
            splits = self._hybrid_split(df, target_column, **kwargs)
        else:
            raise ValueError(f"Unknown splitting strategy: {strategy}")
        
        # Validate splits
        self._validate_splits(splits, df, target_column)
        
        # Save splits if configured
        if self.config["output_format"]["save_splits"]:
            self._save_splits(splits, strategy, **kwargs)
        
        # Record split history
        self._record_split_history(splits, strategy, df.shape)
        
        return splits
    
    def _simple_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Simple random or stratified split"""
        config = self.config["splitting_strategies"]["simple_split"]
        
        # Override with kwargs
        test_size = kwargs.get('test_size', config['test_size'])
        validation_size = kwargs.get('validation_size', config['validation_size'])
        random_state = kwargs.get('random_state', config['random_state'])
        stratify = kwargs.get('stratify', config['stratify'])
        shuffle = kwargs.get('shuffle', config['shuffle'])
        
        # Calculate validation size relative to remaining data after test split
        if validation_size > 0:
            # First split: train+val vs test
            if stratify and df[target_column].nunique() > 1:
                stratify_col = df[target_column]
            else:
                stratify_col = None
            
            train_val_idx, test_idx = train_test_split(
                df.index,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_col,
                shuffle=shuffle
            )
            
            # Second split: train vs validation
            train_val_df = df.loc[train_val_idx]
            if stratify and train_val_df[target_column].nunique() > 1:
                stratify_col_val = train_val_df[target_column]
            else:
                stratify_col_val = None
            
            train_idx, val_idx = train_test_split(
                train_val_df.index,
                test_size=validation_size/(1-test_size),  # Adjust for previous split
                random_state=random_state,
                stratify=stratify_col_val,
                shuffle=shuffle
            )
        else:
            # Only train-test split
            if stratify and df[target_column].nunique() > 1:
                stratify_col = df[target_column]
            else:
                stratify_col = None
            
            train_idx, test_idx = train_test_split(
                df.index,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_col,
                shuffle=shuffle
            )
            val_idx = pd.Index([])
        
        # Create splits dictionary
        splits = self._create_splits_dict(df, train_idx, val_idx, test_idx, target_column)
        
        # Add metadata
        splits["metadata"]["strategy"] = "simple_split"
        splits["metadata"]["test_size"] = test_size
        splits["metadata"]["validation_size"] = validation_size
        splits["metadata"]["random_state"] = random_state
        splits["metadata"]["stratify"] = stratify
        splits["metadata"]["shuffle"] = shuffle
        
        logger.info(f"Simple split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        return splits
    
    def _time_based_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Time-based chronological split"""
        config = self.config["splitting_strategies"]["time_based_split"]
        
        # Override with kwargs
        date_column = kwargs.get('date_column', config['date_column'])
        test_time_period = kwargs.get('test_time_period', config['test_time_period'])
        validation_time_period = kwargs.get('validation_time_period', config['validation_time_period'])
        chronological = kwargs.get('chronological', config['chronological'])
        
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        
        # Ensure date column is datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Sort by date
        df = df.sort_values(date_column)
        
        # Calculate split indices based on time periods
        if chronological:
            # Chronological split: oldest -> newest
            total_days = (df[date_column].max() - df[date_column].min()).days
            
            # Convert time periods to days
            test_days = self._parse_time_period(test_time_period)
            val_days = self._parse_time_period(validation_time_period)
            
            # Calculate split points
            cutoff_test = df[date_column].max() - pd.Timedelta(days=test_days)
            cutoff_val = cutoff_test - pd.Timedelta(days=val_days)
            
            # Create splits
            train_idx = df[df[date_column] <= cutoff_val].index
            val_idx = df[(df[date_column] > cutoff_val) & (df[date_column] <= cutoff_test)].index
            test_idx = df[df[date_column] > cutoff_test].index
            
        else:
            # Random time-based split (preserving time order within splits)
            # First split temporally, then randomize within time periods
            unique_dates = df[date_column].dt.date.unique()
            n_dates = len(unique_dates)
            
            # Split dates
            test_date_count = int(n_dates * 0.2)  # 20% of dates for test
            val_date_count = int(n_dates * 0.1)   # 10% of dates for validation
            
            # Sort dates and split
            sorted_dates = np.sort(unique_dates)
            train_dates = sorted_dates[:-(test_date_count + val_date_count)]
            val_dates = sorted_dates[-(test_date_count + val_date_count):-test_date_count]
            test_dates = sorted_dates[-test_date_count:]
            
            # Get indices for each date group
            train_idx = df[df[date_column].dt.date.isin(train_dates)].index
            val_idx = df[df[date_column].dt.date.isin(val_dates)].index
            test_idx = df[df[date_column].dt.date.isin(test_dates)].index
        
        # Create splits dictionary
        splits = self._create_splits_dict(df, train_idx, val_idx, test_idx, target_column)
        
        # Add time-based metadata
        splits["metadata"]["strategy"] = "time_based_split"
        splits["metadata"]["date_column"] = date_column
        splits["metadata"]["test_time_period"] = test_time_period
        splits["metadata"]["validation_time_period"] = validation_time_period
        splits["metadata"]["chronological"] = chronological
        
        if len(train_idx) > 0:
            splits["metadata"]["train_date_range"] = {
                "start": df.loc[train_idx, date_column].min().isoformat(),
                "end": df.loc[train_idx, date_column].max().isoformat()
            }
        if len(val_idx) > 0:
            splits["metadata"]["validation_date_range"] = {
                "start": df.loc[val_idx, date_column].min().isoformat(),
                "end": df.loc[val_idx, date_column].max().isoformat()
            }
        if len(test_idx) > 0:
            splits["metadata"]["test_date_range"] = {
                "start": df.loc[test_idx, date_column].min().isoformat(),
                "end": df.loc[test_idx, date_column].max().isoformat()
            }
        
        logger.info(f"Time-based split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        return splits
    
    def _cross_validation_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Create cross-validation splits"""
        config = self.config["splitting_strategies"]["cross_validation"]
        
        # Override with kwargs
        n_splits = kwargs.get('n_splits', config['n_splits'])
        random_state = kwargs.get('random_state', config['random_state'])
        shuffle = kwargs.get('shuffle', config['shuffle'])
        stratify = kwargs.get('stratify', config['stratify'])
        
        # Initialize CV splitter
        if stratify and df[target_column].nunique() > 1:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
            y = df[target_column]
        else:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
            y = None
        
        # Generate folds
        folds = []
        X_indices = np.arange(len(df))
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_indices, y)):
            fold = {
                "fold": fold_idx,
                "train_indices": df.index[train_idx].tolist(),
                "test_indices": df.index[test_idx].tolist(),
                "validation_indices": []  # No validation in CV
            }
            folds.append(fold)
        
        # For CV, we don't have fixed train/val/test, just folds
        splits = {
            "strategy": "cross_validation",
            "folds": folds,
            "metadata": {
                "n_splits": n_splits,
                "random_state": random_state,
                "shuffle": shuffle,
                "stratify": stratify,
                "created_at": datetime.now().isoformat()
            }
        }
        
        logger.info(f"Created {n_splits}-fold cross-validation splits")
        return splits
    
    def _domain_based_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Split based on domains/sources to prevent leakage"""
        config = self.config["splitting_strategies"]["domain_based_split"]
        
        # Override with kwargs
        domain_column = kwargs.get('domain_column', config['domain_column'])
        test_domain_ratio = kwargs.get('test_domain_ratio', config['test_domain_ratio'])
        holdout_domains = kwargs.get('holdout_domains', config['holdout_domains'])
        
        if domain_column not in df.columns:
            raise ValueError(f"Domain column '{domain_column}' not found in DataFrame")
        
        # Get unique domains
        domains = df[domain_column].dropna().unique()
        n_domains = len(domains)
        
        if n_domains < 2:
            raise ValueError(f"Need at least 2 unique domains for domain-based split. Found {n_domains}")
        
        # Calculate number of domains for test set
        n_test_domains = max(1, int(n_domains * test_domain_ratio))
        
        # Separate holdout domains if specified
        available_domains = [d for d in domains if d not in holdout_domains]
        
        if len(available_domains) < n_test_domains:
            raise ValueError(f"Not enough available domains after removing holdouts")
        
        # Randomly select test domains
        np.random.seed(kwargs.get('random_state', 42))
        test_domains = np.random.choice(available_domains, size=n_test_domains, replace=False)
        train_val_domains = [d for d in available_domains if d not in test_domains]
        
        # Split train_val_domains into train and validation
        n_val_domains = max(1, int(len(train_val_domains) * 0.2))  # 20% for validation
        val_domains = np.random.choice(train_val_domains, size=n_val_domains, replace=False)
        train_domains = [d for d in train_val_domains if d not in val_domains]
        
        # Get indices for each domain group
        train_idx = df[df[domain_column].isin(train_domains)].index
        val_idx = df[df[domain_column].isin(val_domains)].index
        test_idx = df[df[domain_column].isin(test_domains)].index
        
        # Add holdout domains to test set if specified
        if holdout_domains:
            holdout_idx = df[df[domain_column].isin(holdout_domains)].index
            test_idx = test_idx.union(holdout_idx)
        
        # Create splits dictionary
        splits = self._create_splits_dict(df, train_idx, val_idx, test_idx, target_column)
        
        # Add domain-based metadata
        splits["metadata"]["strategy"] = "domain_based_split"
        splits["metadata"]["domain_column"] = domain_column
        splits["metadata"]["test_domain_ratio"] = test_domain_ratio
        splits["metadata"]["test_domains"] = test_domains.tolist() if isinstance(test_domains, np.ndarray) else test_domains
        splits["metadata"]["validation_domains"] = val_domains.tolist() if isinstance(val_domains, np.ndarray) else val_domains
        splits["metadata"]["train_domains"] = train_domains
        splits["metadata"]["holdout_domains"] = holdout_domains
        
        logger.info(f"Domain-based split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        logger.info(f"Test domains: {len(test_domains)}, Validation domains: {len(val_domains)}, Train domains: {len(train_domains)}")
        
        return splits
    
    def _author_based_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Split based on authors to prevent author leakage"""
        config = self.config["splitting_strategies"]["author_based_split"]
        
        # Override with kwargs
        author_column = kwargs.get('author_column', config['author_column'])
        test_author_ratio = kwargs.get('test_author_ratio', config['test_author_ratio'])
        holdout_authors = kwargs.get('holdout_authors', config['holdout_authors'])
        
        if author_column not in df.columns:
            raise ValueError(f"Author column '{author_column}' not found in DataFrame")
        
        # Get unique authors
        authors = df[author_column].dropna().unique()
        n_authors = len(authors)
        
        if n_authors < 2:
            raise ValueError(f"Need at least 2 unique authors for author-based split. Found {n_authors}")
        
        # Calculate number of authors for test set
        n_test_authors = max(1, int(n_authors * test_author_ratio))
        
        # Separate holdout authors if specified
        available_authors = [a for a in authors if a not in holdout_authors]
        
        if len(available_authors) < n_test_authors:
            raise ValueError(f"Not enough available authors after removing holdouts")
        
        # Randomly select test authors
        np.random.seed(kwargs.get('random_state', 42))
        test_authors = np.random.choice(available_authors, size=n_test_authors, replace=False)
        train_val_authors = [a for a in available_authors if a not in test_authors]
        
        # Split train_val_authors into train and validation
        n_val_authors = max(1, int(len(train_val_authors) * 0.2))  # 20% for validation
        val_authors = np.random.choice(train_val_authors, size=n_val_authors, replace=False)
        train_authors = [a for a in train_val_authors if a not in val_authors]
        
        # Get indices for each author group
        train_idx = df[df[author_column].isin(train_authors)].index
        val_idx = df[df[author_column].isin(val_authors)].index
        test_idx = df[df[author_column].isin(test_authors)].index
        
        # Add holdout authors to test set if specified
        if holdout_authors:
            holdout_idx = df[df[author_column].isin(holdout_authors)].index
            test_idx = test_idx.union(holdout_idx)
        
        # Create splits dictionary
        splits = self._create_splits_dict(df, train_idx, val_idx, test_idx, target_column)
        
        # Add author-based metadata
        splits["metadata"]["strategy"] = "author_based_split"
        splits["metadata"]["author_column"] = author_column
        splits["metadata"]["test_author_ratio"] = test_author_ratio
        splits["metadata"]["test_authors"] = test_authors.tolist() if isinstance(test_authors, np.ndarray) else test_authors
        splits["metadata"]["validation_authors"] = val_authors.tolist() if isinstance(val_authors, np.ndarray) else val_authors
        splits["metadata"]["train_authors"] = train_authors
        splits["metadata"]["holdout_authors"] = holdout_authors
        
        logger.info(f"Author-based split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        logger.info(f"Test authors: {len(test_authors)}, Validation authors: {len(val_authors)}, Train authors: {len(train_authors)}")
        
        return splits
    
    def _hybrid_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Hybrid splitting strategy combining multiple approaches"""
        # Example: Time-based split for test, random split for validation
        test_strategy = kwargs.get('test_strategy', 'time_based')
        val_strategy = kwargs.get('val_strategy', 'simple')
        
        logger.info(f"Hybrid split: Test={test_strategy}, Validation={val_strategy}")
        
        if test_strategy == 'time_based':
            # Use time-based for test split
            time_config = self.config["splitting_strategies"]["time_based_split"]
            date_column = kwargs.get('date_column', time_config['date_column'])
            
            if date_column not in df.columns:
                raise ValueError(f"Date column '{date_column}' not found for time-based test split")
            
            # Sort by date
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.sort_values(date_column)
            
            # Split chronologically (80% train+val, 20% test)
            split_point = int(len(df) * 0.8)
            train_val_df = df.iloc[:split_point]
            test_df = df.iloc[split_point:]
            
            test_idx = test_df.index
            
            # Use simple split for train/validation
            simple_config = self.config["splitting_strategies"]["simple_split"]
            val_size = kwargs.get('validation_size', simple_config['validation_size'])
            random_state = kwargs.get('random_state', simple_config['random_state'])
            
            if val_size > 0:
                # Split train_val_df into train and validation
                if simple_config['stratify'] and train_val_df[target_column].nunique() > 1:
                    stratify_col = train_val_df[target_column]
                else:
                    stratify_col = None
                
                train_idx, val_idx = train_test_split(
                    train_val_df.index,
                    test_size=val_size,
                    random_state=random_state,
                    stratify=stratify_col,
                    shuffle=True
                )
            else:
                train_idx = train_val_df.index
                val_idx = pd.Index([])
        
        else:
            # Default to simple split for both
            return self._simple_split(df, target_column, **kwargs)
        
        # Create splits dictionary
        splits = self._create_splits_dict(df, train_idx, val_idx, test_idx, target_column)
        
        # Add hybrid metadata
        splits["metadata"]["strategy"] = "hybrid_split"
        splits["metadata"]["test_strategy"] = test_strategy
        splits["metadata"]["val_strategy"] = val_strategy
        
        logger.info(f"Hybrid split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        return splits
    
    def _create_splits_dict(
        self,
        df: pd.DataFrame,
        train_idx: pd.Index,
        val_idx: pd.Index,
        test_idx: pd.Index,
        target_column: str
    ) -> Dict[str, Any]:
        """Create standardized splits dictionary"""
        # Get data for each split
        train_df = df.loc[train_idx] if len(train_idx) > 0 else pd.DataFrame()
        val_df = df.loc[val_idx] if len(val_idx) > 0 else pd.DataFrame()
        test_df = df.loc[test_idx] if len(test_idx) > 0 else pd.DataFrame()
        
        # Calculate class distributions
        train_dist = self._calculate_class_distribution(train_df, target_column) if len(train_df) > 0 else {}
        val_dist = self._calculate_class_distribution(val_df, target_column) if len(val_df) > 0 else {}
        test_dist = self._calculate_class_distribution(test_df, target_column) if len(test_df) > 0 else {}
        
        splits = {
            "train": {
                "indices": train_idx.tolist(),
                "dataframe": train_df,
                "size": len(train_idx),
                "class_distribution": train_dist
            },
            "validation": {
                "indices": val_idx.tolist(),
                "dataframe": val_df,
                "size": len(val_idx),
                "class_distribution": val_dist
            },
            "test": {
                "indices": test_idx.tolist(),
                "dataframe": test_df,
                "size": len(test_idx),
                "class_distribution": test_dist
            },
            "metadata": {
                "total_samples": len(df),
                "created_at": datetime.now().isoformat(),
                "train_ratio": len(train_idx) / len(df) if len(df) > 0 else 0,
                "validation_ratio": len(val_idx) / len(df) if len(df) > 0 else 0,
                "test_ratio": len(test_idx) / len(df) if len(df) > 0 else 0,
                "target_column": target_column
            }
        }
        
        return splits
    
    def _calculate_class_distribution(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Calculate class distribution for a DataFrame"""
        if target_column not in df.columns or df.empty:
            return {}
        
        counts = df[target_column].value_counts()
        percentages = (counts / len(df) * 100).round(2)
        
        return {
            "counts": counts.to_dict(),
            "percentages": percentages.to_dict(),
            "imbalance_ratio": counts.min() / counts.max() if len(counts) > 1 else 1.0
        }
    
    def _parse_time_period(self, time_period: str) -> int:
        """Parse time period string to days"""
        if time_period.endswith('D'):
            return int(time_period[:-1])
        elif time_period.endswith('W'):
            return int(time_period[:-1]) * 7
        elif time_period.endswith('M'):
            return int(time_period[:-1]) * 30
        elif time_period.endswith('Y'):
            return int(time_period[:-1]) * 365
        else:
            return int(time_period)  # Assume days
    
    def _validate_split_input(
        self,
        df: pd.DataFrame,
        target_column: str,
        strategy: str,
        **kwargs
    ):
        """Validate input parameters for splitting"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Check minimum samples
        min_samples = self.config["data_requirements"]["min_samples_per_split"]
        if len(df) < min_samples:
            logger.warning(f"Dataset has only {len(df)} samples, minimum recommended is {min_samples}")
        
        # Check class imbalance
        if self.config["data_requirements"]["ensure_all_classes"]:
            unique_classes = df[target_column].nunique()
            if unique_classes < 2:
                raise ValueError(f"Need at least 2 classes for splitting. Found {unique_classes}")
            
            class_counts = df[target_column].value_counts()
            min_count = class_counts.min()
            max_count = class_counts.max()
            imbalance = min_count / max_count
            
            max_imbalance = self.config["data_requirements"]["max_class_imbalance"]
            if imbalance < max_imbalance:
                logger.warning(f"High class imbalance detected: ratio={imbalance:.3f}")
        
        # Strategy-specific validations
        if strategy == "time_based_split":
            date_column = kwargs.get('date_column', self.config["splitting_strategies"]["time_based_split"]["date_column"])
            if date_column not in df.columns:
                raise ValueError(f"Date column '{date_column}' required for time-based split")
        
        elif strategy == "domain_based_split":
            domain_column = kwargs.get('domain_column', self.config["splitting_strategies"]["domain_based_split"]["domain_column"])
            if domain_column not in df.columns:
                raise ValueError(f"Domain column '{domain_column}' required for domain-based split")
        
        elif strategy == "author_based_split":
            author_column = kwargs.get('author_column', self.config["splitting_strategies"]["author_based_split"]["author_column"])
            if author_column not in df.columns:
                raise ValueError(f"Author column '{author_column}' required for author-based split")
    
    def _validate_splits(self, splits: Dict, original_df: pd.DataFrame, target_column: str):
        """Validate that splits meet requirements"""
        requirements = self.config["data_requirements"]
        
        # Check for empty splits
        for split_name in ["train", "validation", "test"]:
            if split_name in splits:
                split_size = splits[split_name]["size"]
                if split_size == 0 and split_name != "validation":  # Validation can be empty
                    logger.warning(f"{split_name.capitalize()} split is empty")
        
        # Check class representation in train split
        if "train" in splits and splits["train"]["size"] > 0:
            train_dist = splits["train"]["class_distribution"]
            if "counts" in train_dist:
                unique_classes = len(train_dist["counts"])
                if unique_classes < 2:
                    logger.warning("Train split has less than 2 classes")
                
                # Check minimum samples per class
                for cls, count in train_dist["counts"].items():
                    if count < requirements["min_samples_per_split"]:
                        logger.warning(f"Class {cls} has only {count} samples in train split")
        
        # Check for data leakage (if indices are available)
        if all(key in splits for key in ["train", "validation", "test"]):
            train_indices = set(splits["train"]["indices"])
            val_indices = set(splits["validation"]["indices"])
            test_indices = set(splits["test"]["indices"])
            
            # Check for overlaps
            if train_indices & val_indices:
                logger.warning("Overlap between train and validation sets")
            if train_indices & test_indices:
                logger.warning("Overlap between train and test sets")
            if val_indices & test_indices:
                logger.warning("Overlap between validation and test sets")
            
            # Check coverage
            all_indices = train_indices | val_indices | test_indices
            original_indices = set(original_df.index)
            missing_indices = original_indices - all_indices
            extra_indices = all_indices - original_indices
            
            if missing_indices:
                logger.warning(f"{len(missing_indices)} indices missing from splits")
            if extra_indices:
                logger.warning(f"{len(extra_indices)} extra indices in splits")
    
    def _save_splits(self, splits: Dict, strategy: str, **kwargs):
        """Save splits to disk"""
        output_config = self.config["output_format"]
        output_dir = Path(output_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save indices if configured
        if output_config["save_indices"]:
            indices_file = output_dir / f"{strategy}_indices_{timestamp}.json"
            indices_data = {
                "train_indices": splits.get("train", {}).get("indices", []),
                "validation_indices": splits.get("validation", {}).get("indices", []),
                "test_indices": splits.get("test", {}).get("indices", [])
            }
            
            with open(indices_file, 'w') as f:
                json.dump(indices_data, f, indent=2)
            
            logger.info(f"Saved split indices to {indices_file}")
        
        # Save metadata if configured
        if output_config["save_metadata"] and "metadata" in splits:
            metadata_file = output_dir / f"{strategy}_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(splits["metadata"], f, indent=2, default=str)
            
            logger.info(f"Saved split metadata to {metadata_file}")
        
        # Save DataFrames if configured
        format_type = output_config["format"]
        
        for split_name in ["train", "validation", "test"]:
            if split_name in splits and splits[split_name]["size"] > 0:
                df = splits[split_name]["dataframe"]
                
                if format_type == "csv":
                    file_path = output_dir / f"{split_name}_{timestamp}.csv"
                    df.to_csv(file_path, index=False)
                elif format_type == "parquet":
                    file_path = output_dir / f"{split_name}_{timestamp}.parquet"
                    df.to_parquet(file_path, index=False)
                elif format_type == "pickle":
                    file_path = output_dir / f"{split_name}_{timestamp}.pkl"
                    df.to_pickle(file_path)
                
                logger.info(f"Saved {split_name} split to {file_path}")
    
    def _record_split_history(self, splits: Dict, strategy: str, original_shape: Tuple):
        """Record split in history"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "original_shape": original_shape,
            "train_size": splits.get("train", {}).get("size", 0),
            "validation_size": splits.get("validation", {}).get("size", 0),
            "test_size": splits.get("test", {}).get("size", 0),
            "metadata": splits.get("metadata", {})
        }
        
        self.split_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.split_history) > 100:
            self.split_history = self.split_history[-100:]
    
    def get_split_history(self) -> pd.DataFrame:
        """Get split history as DataFrame"""
        return pd.DataFrame(self.split_history)
    
    def create_multiple_splits(
        self,
        df: pd.DataFrame,
        target_column: str,
        strategies: List[str] = None,
        n_repeats: int = 3,
        **kwargs
    ) -> Dict[str, List[Dict]]:
        """
        Create multiple splits using different strategies
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            strategies: List of strategies to use
            n_repeats: Number of repeats for each strategy
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with splits for each strategy
        """
        if strategies is None:
            strategies = ["simple_split", "time_based_split", "domain_based_split"]
        
        all_splits = {}
        
        for strategy in strategies:
            strategy_splits = []
            
            for repeat in range(n_repeats):
                logger.info(f"Creating split {repeat+1}/{n_repeats} with {strategy}")
                
                # Adjust random seed for each repeat
                if "random_state" in kwargs:
                    kwargs["random_state"] = kwargs["random_state"] + repeat
                else:
                    kwargs["random_state"] = 42 + repeat
                
                try:
                    splits = self.split_data(df, target_column, strategy, **kwargs)
                    splits["metadata"]["repeat"] = repeat
                    strategy_splits.append(splits)
                except Exception as e:
                    logger.error(f"Error creating split with {strategy}: {e}")
            
            all_splits[strategy] = strategy_splits
        
        return all_splits
    
    def analyze_splits(self, splits: Dict) -> pd.DataFrame:
        """
        Analyze and compare different splits
        
        Args:
            splits: Dictionary of splits from create_multiple_splits
        
        Returns:
            DataFrame with split analysis
        """
        analysis_rows = []
        
        for strategy, strategy_splits in splits.items():
            for i, split in enumerate(strategy_splits):
                row = {
                    "strategy": strategy,
                    "split_id": i,
                    "train_size": split["train"]["size"],
                    "validation_size": split.get("validation", {}).get("size", 0),
                    "test_size": split["test"]["size"],
                    "train_ratio": split["metadata"]["train_ratio"],
                    "test_ratio": split["metadata"]["test_ratio"]
                }
                
                # Add class distribution metrics
                if "class_distribution" in split["train"]:
                    train_dist = split["train"]["class_distribution"]
                    if "imbalance_ratio" in train_dist:
                        row["train_imbalance"] = train_dist["imbalance_ratio"]
                
                analysis_rows.append(row)
        
        return pd.DataFrame(analysis_rows)
