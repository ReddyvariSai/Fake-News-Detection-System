import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation and quality assurance for fake news datasets"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.validation_history = []
    
    def _get_default_config(self) -> Dict:
        """Get default validation configuration"""
        return {
            "quality_thresholds": {
                "missing_values_percentage": 30.0,
                "class_imbalance_ratio": 0.3,
                "min_samples_per_class": 100,
                "text_length_min": 10,
                "text_length_max": 10000,
                "duplicate_threshold": 0.05,
                "outlier_threshold": 3.0  # z-score threshold
            },
            "validation_rules": {
                "check_missing_values": True,
                "check_class_distribution": True,
                "check_text_quality": True,
                "check_data_types": True,
                "check_duplicates": True,
                "check_outliers": True,
                "check_feature_correlation": True,
                "check_label_consistency": True,
                "check_date_validity": True,
                "check_source_domains": True
            },
            "text_validation": {
                "min_words": 5,
                "max_words": 1000,
                "allowed_languages": ["english"],
                "invalid_patterns": [
                    r"test.*",
                    r"sample.*",
                    r"placeholder.*",
                    r"lorem ipsum"
                ]
            }
        }
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        label_column: Optional[str] = None,
        text_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive dataset validation
        
        Args:
            df: Input DataFrame
            dataset_name: Name of dataset for reporting
            label_column: Name of label column
            text_column: Name of text column
        
        Returns:
            Validation results
        """
        logger.info(f"Starting validation for {dataset_name}...")
        
        validation_results = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "basic_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "warnings": 0,
                "overall_status": "PASS"
            },
            "checks": [],
            "issues": [],
            "recommendations": [],
            "metrics": {}
        }
        
        # Basic data structure check
        validation_results["checks"].append(
            self._check_data_structure(df, dataset_name)
        )
        
        # Missing values check
        if self.config["validation_rules"]["check_missing_values"]:
            validation_results["checks"].append(
                self._check_missing_values(df, dataset_name)
            )
        
        # Data types check
        if self.config["validation_rules"]["check_data_types"]:
            validation_results["checks"].append(
                self._check_data_types(df, dataset_name)
            )
        
        # Duplicates check
        if self.config["validation_rules"]["check_duplicates"]:
            validation_results["checks"].append(
                self._check_duplicates(df, dataset_name, text_column)
            )
        
        # Class distribution check (if label column provided)
        if (self.config["validation_rules"]["check_class_distribution"] and 
            label_column and label_column in df.columns):
            validation_results["checks"].append(
                self._check_class_distribution(df, dataset_name, label_column)
            )
        
        # Text quality check (if text column provided)
        if (self.config["validation_rules"]["check_text_quality"] and 
            text_column and text_column in df.columns):
            validation_results["checks"].append(
                self._check_text_quality(df, dataset_name, text_column)
            )
        
        # Outliers check
        if self.config["validation_rules"]["check_outliers"]:
            validation_results["checks"].append(
                self._check_outliers(df, dataset_name)
            )
        
        # Feature correlation check
        if self.config["validation_rules"]["check_feature_correlation"]:
            validation_results["checks"].append(
                self._check_feature_correlation(df, dataset_name)
            )
        
        # Label consistency check (for binary classification)
        if (self.config["validation_rules"]["check_label_consistency"] and 
            label_column and label_column in df.columns):
            validation_results["checks"].append(
                self._check_label_consistency(df, dataset_name, label_column)
            )
        
        # Date validity check
        if self.config["validation_rules"]["check_date_validity"]:
            validation_results["checks"].append(
                self._check_date_validity(df, dataset_name)
            )
        
        # Source domains check
        if self.config["validation_rules"]["check_source_domains"]:
            validation_results["checks"].append(
                self._check_source_domains(df, dataset_name)
            )
        
        # Calculate validation metrics
        validation_results["metrics"] = self._calculate_validation_metrics(df)
        
        # Summarize results
        self._summarize_validation(validation_results)
        
        # Generate recommendations
        validation_results["recommendations"] = self._generate_recommendations(
            validation_results["checks"]
        )
        
        # Save validation history
        self.validation_history.append({
            "timestamp": validation_results["timestamp"],
            "dataset_name": dataset_name,
            "summary": validation_results["summary"],
            "rows": len(df),
            "columns": len(df.columns)
        })
        
        # Save validation report
        self._save_validation_report(validation_results, dataset_name)
        
        logger.info(f"Validation completed for {dataset_name}")
        logger.info(f"Overall status: {validation_results['summary']['overall_status']}")
        
        return validation_results
    
    def _check_data_structure(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check basic data structure"""
        check_name = "Data Structure"
        results = {
            "check_name": check_name,
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Check if DataFrame is empty
            if df.empty:
                results["status"] = "FAIL"
                results["issues"].append("DataFrame is empty")
                return results
            
            # Check dimensions
            results["details"]["rows"] = len(df)
            results["details"]["columns"] = len(df.columns)
            
            # Check for required columns (basic)
            required_columns = ["text", "label"]
            missing_required = [col for col in required_columns if col not in df.columns]
            
            if missing_required:
                results["status"] = "WARNING"
                results["issues"].append(f"Missing recommended columns: {missing_required}")
            
            # Check column names for consistency
            invalid_chars = []
            for col in df.columns:
                if any(char in col for char in [' ', '-', '.', '/']):
                    invalid_chars.append(col)
            
            if invalid_chars:
                results["status"] = "WARNING"
                results["issues"].append(f"Columns with special characters: {invalid_chars}")
            
            logger.info(f"{check_name}: {results['status']}")
            return results
            
        except Exception as e:
            results["status"] = "ERROR"
            results["issues"].append(f"Error in structure check: {str(e)}")
            return results
    
    def _check_missing_values(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check for missing values"""
        check_name = "Missing Values"
        threshold = self.config["quality_thresholds"]["missing_values_percentage"]
        
        results = {
            "check_name": check_name,
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Calculate missing values
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100
            
            # Identify columns with missing values
            columns_with_missing = missing_counts[missing_counts > 0]
            
            if not columns_with_missing.empty:
                results["details"]["missing_summary"] = {
                    "columns_with_missing": len(columns_with_missing),
                    "total_missing": int(missing_counts.sum()),
                    "overall_missing_percentage": float((missing_counts.sum() / (len(df) * len(df.columns))) * 100)
                }
                
                # Check each column against threshold
                problematic_columns = []
                for col, missing_pct in missing_percentages.items():
                    if missing_pct > threshold:
                        problematic_columns.append({
                            "column": col,
                            "missing_count": int(missing_counts[col]),
                            "missing_percentage": float(missing_pct)
                        })
                
                if problematic_columns:
                    results["status"] = "FAIL"
                    results["issues"].append(f"Columns exceed {threshold}% missing threshold")
                    results["details"]["problematic_columns"] = problematic_columns
                else:
                    results["status"] = "WARNING"
                    results["issues"].append("Missing values detected but within threshold")
            
            logger.info(f"{check_name}: {results['status']}")
            return results
            
        except Exception as e:
            results["status"] = "ERROR"
            results["issues"].append(f"Error in missing values check: {str(e)}")
            return results
    
    def _check_data_types(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check data types consistency"""
        check_name = "Data Types"
        
        results = {
            "check_name": check_name,
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Get data types
            dtypes = df.dtypes.astype(str).to_dict()
            results["details"]["data_types"] = dtypes
            
            # Check for object columns that should be numeric
            numeric_columns = []
            for col, dtype in dtypes.items():
                if dtype == 'object':
                    # Try to convert to numeric
                    try:
                        pd.to_numeric(df[col].dropna())
                        numeric_columns.append(col)
                    except:
                        pass
            
            if numeric_columns:
                results["status"] = "WARNING"
                results["issues"].append(f"Columns that could be numeric: {numeric_columns}")
                results["details"]["potential_numeric_columns"] = numeric_columns
            
            # Check text columns for proper encoding
            text_columns = df.select_dtypes(include=['object']).columns
            encoding_issues = []
            
            for col in text_columns:
                sample = df[col].dropna().head(100)
                if not sample.empty:
                    # Check for encoding issues
                    try:
                        sample.astype(str).str.encode('utf-8')
                    except UnicodeEncodeError:
                        encoding_issues.append(col)
            
            if encoding_issues:
                results["status"] = "WARNING"
                results["issues"].append(f"Encoding issues in columns: {encoding_issues}")
            
            logger.info(f"{check_name}: {results['status']}")
            return results
            
        except Exception as e:
            results["status"] = "ERROR"
            results["issues"].append(f"Error in data types check: {str(e)}")
            return results
    
    def _check_duplicates(
        self, 
        df: pd.DataFrame, 
        dataset_name: str, 
        text_column: Optional[str] = None
    ) -> Dict:
        """Check for duplicate records"""
        check_name = "Duplicate Records"
        threshold = self.config["quality_thresholds"]["duplicate_threshold"]
        
        results = {
            "check_name": check_name,
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Check exact duplicates across all columns
            exact_duplicates = df.duplicated().sum()
            duplicate_percentage = (exact_duplicates / len(df)) * 100
            
            results["details"]["exact_duplicates"] = {
                "count": int(exact_duplicates),
                "percentage": float(duplicate_percentage)
            }
            
            if duplicate_percentage > (threshold * 100):
                results["status"] = "FAIL"
                results["issues"].append(f"High percentage of exact duplicates: {duplicate_percentage:.1f}%")
            
            # Check for near-duplicates in text column
            if text_column and text_column in df.columns:
                # Simple text similarity check
                text_duplicates = self._find_text_duplicates(df[text_column])
                text_duplicate_percentage = (len(text_duplicates) / len(df)) * 100
                
                results["details"]["text_duplicates"] = {
                    "count": len(text_duplicates),
                    "percentage": float(text_duplicate_percentage),
                    "examples": text_duplicates[:5]  # Show first 5 examples
                }
                
                if text_duplicate_percentage > (threshold * 100):
                    if results["status"] == "PASS":
                        results["status"] = "WARNING"
                    results["issues"].append(f"Possible near-duplicates in text: {text_duplicate_percentage:.1f}%")
            
            logger.info(f"{check_name}: {results['status']}")
            return results
            
        except Exception as e:
            results["status"] = "ERROR"
            results["issues"].append(f"Error in duplicate check: {str(e)}")
            return results
    
    def _check_class_distribution(
        self, 
        df: pd.DataFrame, 
        dataset_name: str, 
        label_column: str
    ) -> Dict:
        """Check class distribution for classification tasks"""
        check_name = "Class Distribution"
        imbalance_threshold = self.config["quality_thresholds"]["class_imbalance_ratio"]
        min_samples = self.config["quality_thresholds"]["min_samples_per_class"]
        
        results = {
            "check_name": check_name,
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Get class distribution
            class_counts = df[label_column].value_counts()
            class_percentages = (class_counts / len(df)) * 100
            
            results["details"]["distribution"] = {
                "counts": class_counts.to_dict(),
                "percentages": class_percentages.to_dict()
            }
            
            # Check for minimum samples per class
            undersampled_classes = []
            for class_name, count in class_counts.items():
                if count < min_samples:
                    undersampled_classes.append({
                        "class": class_name,
                        "count": int(count),
                        "minimum_required": min_samples
                    })
            
            if undersampled_classes:
                results["status"] = "FAIL"
                results["issues"].append(f"Classes with insufficient samples: {undersampled_classes}")
            
            # Check for class imbalance
            if len(class_counts) >= 2:
                max_count = class_counts.max()
                min_count = class_counts.min()
                imbalance_ratio = min_count / max_count
                
                results["details"]["imbalance"] = {
                    "ratio": float(imbalance_ratio),
                    "max_class": class_counts.idxmax(),
                    "min_class": class_counts.idxmin()
                }
                
                if imbalance_ratio < imbalance_threshold:
                    results["status"] = "WARNING"
                    results["issues"].append(f"Class imbalance detected: ratio = {imbalance_ratio:.3f}")
            
            logger.info(f"{check_name}: {results['status']}")
            return results
            
        except Exception as e:
            results["status"] = "ERROR"
            results["issues"].append(f"Error in class distribution check: {str(e)}")
            return results
    
    def _check_text_quality(
        self, 
        df: pd.DataFrame, 
        dataset_name: str, 
        text_column: str
    ) -> Dict:
        """Check text quality and content"""
        check_name = "Text Quality"
        min_words = self.config["text_validation"]["min_words"]
        max_words = self.config["text_validation"]["max_words"]
        
        results = {
            "check_name": check_name,
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Calculate text statistics
            text_stats = df[text_column].apply(self._analyze_text_quality)
            stats_df = pd.DataFrame(text_stats.tolist())
            
            # Identify problematic texts
            short_texts = stats_df[stats_df['word_count'] < min_words]
            long_texts = stats_df[stats_df['word_count'] > max_words]
            empty_texts = df[text_column].isna() | (df[text_column].str.strip() == "")
            
            # Check for invalid patterns
            invalid_patterns = self.config["text_validation"]["invalid_patterns"]
            pattern_matches = []
            
            for pattern in invalid_patterns:
                matches = df[text_column].str.contains(pattern, case=False, na=False)
                if matches.any():
                    pattern_matches.append({
                        "pattern": pattern,
                        "count": int(matches.sum()),
                        "examples": df[text_column][matches].head(3).tolist()
                    })
            
            # Compile results
            results["details"]["statistics"] = {
                "average_word_count": float(stats_df['word_count'].mean()),
                "average_char_count": float(stats_df['char_count'].mean()),
                "short_texts": {
                    "count": len(short_texts),
                    "percentage": (len(short_texts) / len(df)) * 100
                },
                "long_texts": {
                    "count": len(long_texts),
                    "percentage": (len(long_texts) / len(df)) * 100
                },
                "empty_texts": {
                    "count": int(empty_texts.sum()),
                    "percentage": (empty_texts.sum() / len(df)) * 100
                }
            }
            
            # Check issues
            issues = []
            
            if len(short_texts) > 0:
                short_pct = (len(short_texts) / len(df)) * 100
                if short_pct > 5:  # More than 5% short texts
                    issues.append(f"High percentage of short texts: {short_pct:.1f}%")
            
            if len(long_texts) > 0:
                long_pct = (len(long_texts) / len(df)) * 100
                if long_pct > 10:  # More than 10% long texts
                    issues.append(f"High percentage of long texts: {long_pct:.1f}%")
            
            if pattern_matches:
                issues.append(f"Texts matching invalid patterns: {len(pattern_matches)} patterns found")
                results["details"]["pattern_matches"] = pattern_matches
            
            if issues:
                results["status"] = "WARNING" if results["status"] == "PASS" else "FAIL"
                results["issues"].extend(issues)
            
            logger.info(f"{check_name}: {results['status']}")
            return results
            
        except Exception as e:
            results["status"] = "ERROR"
            results["issues"].append(f"Error in text quality check: {str(e)}")
            return results
    
    def _check_outliers(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check for outliers in numerical columns"""
        check_name = "Outliers"
        threshold = self.config["quality_thresholds"]["outlier_threshold"]
        
        results = {
            "check_name": check_name,
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Get numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                results["details"]["message"] = "No numerical columns found"
                return results
            
            outliers_summary = {}
            
            for col in numeric_cols:
                # Remove NaN values
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    # Calculate z-scores
                    mean = col_data.mean()
                    std = col_data.std()
                    
                    if std > 0:  # Avoid division by zero
                        z_scores = np.abs((col_data - mean) / std)
                        outliers = z_scores > threshold
                        
                        if outliers.any():
                            outliers_summary[col] = {
                                "count": int(outliers.sum()),
                                "percentage": (outliers.sum() / len(col_data)) * 100,
                                "mean": float(mean),
                                "std": float(std)
                            }
            
            results["details"]["outliers"] = outliers_summary
            
            # Check for severe outliers
            severe_outliers = []
            for col, stats in outliers_summary.items():
                if stats["percentage"] > 5:  # More than 5% outliers
                    severe_outliers.append({
                        "column": col,
                        "outlier_percentage": stats["percentage"]
                    })
            
            if severe_outliers:
                results["status"] = "WARNING"
                results["issues"].append(f"High percentage of outliers in columns: {severe_outliers}")
            
            logger.info(f"{check_name}: {results['status']}")
            return results
            
        except Exception as e:
            results["status"] = "ERROR"
            results["issues"].append(f"Error in outlier check: {str(e)}")
            return results
    
    def _check_feature_correlation(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check for high correlations between features"""
        check_name = "Feature Correlation"
        
        results = {
            "check_name": check_name,
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Get numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                results["details"]["message"] = "Insufficient numerical columns for correlation analysis"
                return results
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > 0.8:  # High correlation threshold
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        high_corr_pairs.append({
                            "pair": f"{col1} - {col2}",
                            "correlation": float(corr_value)
                        })
            
            results["details"]["correlation_analysis"] = {
                "total_numeric_columns": len(numeric_cols),
                "highly_correlated_pairs": high_corr_pairs,
                "high_correlation_threshold": 0.8
            }
            
            if high_corr_pairs:
                results["status"] = "WARNING"
                results["issues"].append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
            
            logger.info(f"{check_name}: {results['status']}")
            return results
            
        except Exception as e:
            results["status"] = "ERROR"
            results["issues"].append(f"Error in correlation check: {str(e)}")
            return results
    
    def _check_label_consistency(
        self, 
        df: pd.DataFrame, 
        dataset_name: str, 
        label_column: str
    ) -> Dict:
        """Check label consistency for classification"""
        check_name = "Label Consistency"
        
        results = {
            "check_name": check_name,
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Check for valid binary labels
            if df[label_column].nunique() == 2:
                # Verify labels are 0/1 or similar
                unique_values = sorted(df[label_column].unique())
                
                if unique_values != [0, 1]:
                    results["status"] = "WARNING"
                    results["issues"].append(f"Binary labels are not 0/1: {unique_values}")
                
                results["details"]["binary_labels"] = {
                    "unique_values": unique_values,
                    "is_standard": unique_values == [0, 1]
                }
            
            # Check for invalid labels (NaN, strings in numeric, etc.)
            invalid_labels = df[df[label_column].isna()]
            if not invalid_labels.empty:
                results["status"] = "FAIL"
                results["issues"].append(f"Found {len(invalid_labels)} NaN labels")
            
            logger.info(f"{check_name}: {results['status']}")
            return results
            
        except Exception as e:
            results["status"] = "ERROR"
            results["issues"].append(f"Error in label consistency check: {str(e)}")
            return results
    
    def _check_date_validity(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check date columns for validity"""
        check_name = "Date Validity"
        
        results = {
            "check_name": check_name,
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Identify potential date columns
            date_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(date_term in col_lower for date_term in ['date', 'time', 'published', 'created']):
                    date_columns.append(col)
            
            if not date_columns:
                results["details"]["message"] = "No date-like columns found"
                return results
            
            date_issues = []
            
            for col in date_columns:
                try:
                    # Try to parse dates
                    parsed_dates = pd.to_datetime(df[col], errors='coerce')
                    invalid_dates = parsed_dates.isna().sum()
                    
                    if invalid_dates > 0:
                        date_issues.append({
                            "column": col,
                            "invalid_count": int(invalid_dates),
                            "invalid_percentage": (invalid_dates / len(df)) * 100
                        })
                    
                    # Check date range
                    valid_dates = parsed_dates.dropna()
                    if len(valid_dates) > 0:
                        date_range = {
                            "min": valid_dates.min().isoformat(),
                            "max": valid_dates.max().isoformat()
                        }
                        results["details"][f"{col}_range"] = date_range
                        
                except Exception as e:
                    date_issues.append({
                        "column": col,
                        "error": str(e)
                    })
            
            results["details"]["date_columns"] = date_columns
            results["details"]["issues"] = date_issues
            
            if date_issues:
                results["status"] = "WARNING"
                results["issues"].append(f"Date parsing issues in {len(date_issues)} columns")
            
            logger.info(f"{check_name}: {results['status']}")
            return results
            
        except Exception as e:
            results["status"] = "ERROR"
            results["issues"].append(f"Error in date validity check: {str(e)}")
            return results
    
    def _check_source_domains(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check source/domain columns for validity"""
        check_name = "Source/Domain Validity"
        
        results = {
            "check_name": check_name,
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Identify source/domain columns
            source_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(source_term in col_lower for source_term in ['source', 'domain', 'url', 'publisher']):
                    source_columns.append(col)
            
            if not source_columns:
                results["details"]["message"] = "No source/domain columns found"
                return results
            
            source_issues = []
            
            for col in source_columns:
                # Check for empty/invalid sources
                empty_sources = df[col].isna() | (df[col].str.strip() == "")
                empty_count = empty_sources.sum()
                
                if empty_count > 0:
                    source_issues.append({
                        "column": col,
                        "empty_count": int(empty_count),
                        "empty_percentage": (empty_count / len(df)) * 100
                    })
                
                # Check for unique sources
                unique_sources = df[col].nunique()
                results["details"][f"{col}_stats"] = {
                    "unique_count": int(unique_sources),
                    "top_sources": df[col].value_counts().head(5).to_dict()
                }
            
            results["details"]["source_columns"] = source_columns
            results["details"]["issues"] = source_issues
            
            if source_issues:
                results["status"] = "WARNING"
                results["issues"].append(f"Source/domain issues in {len(source_issues)} columns")
            
            logger.info(f"{check_name}: {results['status']}")
            return results
            
        except Exception as e:
            results["status"] = "ERROR"
            results["issues"].append(f"Error in source/domain check: {str(e)}")
            return results
    
    def _analyze_text_quality(self, text: str) -> Dict:
        """Analyze text quality metrics"""
        if not isinstance(text, str) or pd.isna(text):
            return {
                "word_count": 0,
                "char_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0,
                "has_html": False
            }
        
        # Basic metrics
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Sentence count (simplified)
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        # Check for HTML
        has_html = bool(re.search(r'<[^>]+>', text))
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "has_html": has_html
        }
    
    def _find_text_duplicates(self, text_series: pd.Series, similarity_threshold: float = 0.9) -> List[int]:
        """Find near-duplicate texts using simple hashing"""
        # Simple approach: hash normalized text
        normalized_texts = text_series.fillna('').astype(str).str.lower().str.strip()
        
        # Remove extra whitespace
        normalized_texts = normalized_texts.str.replace(r'\s+', ' ', regex=True)
        
        # Find exact duplicates after normalization
        duplicates = normalized_texts.duplicated(keep=False)
        
        return text_series[duplicates].index.tolist()
    
    def _calculate_validation_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate overall validation metrics"""
        metrics = {
            "data_quality_score": 100.0,
            "completeness_score": 0.0,
            "consistency_score": 0.0,
            "validity_score": 0.0
        }
        
        try:
            # Completeness (based on missing values)
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            completeness = ((total_cells - missing_cells) / total_cells) * 100
            metrics["completeness_score"] = float(completeness)
            
            # Consistency (based on data types and patterns)
            # Simplified calculation
            numeric_consistency = 0
            text_consistency = 0
            
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check for infinite values
                    if np.isfinite(df[col].dropna()).all():
                        numeric_consistency += 1
                elif pd.api.types.is_string_dtype(df[col]):
                    # Check for consistent encoding
                    try:
                        df[col].astype(str).str.encode('utf-8')
                        text_consistency += 1
                    except:
                        pass
            
            total_columns = len(df.columns)
            consistency = ((numeric_consistency + text_consistency) / (2 * total_columns)) * 100
            metrics["consistency_score"] = float(consistency)
            
            # Validity (based on basic constraints)
            # Simplified - could be enhanced based on domain rules
            valid_columns = 0
            for col in df.columns:
                # Basic validity check: column has at least some non-null values
                if df[col].notna().any():
                    valid_columns += 1
            
            validity = (valid_columns / total_columns) * 100
            metrics["validity_score"] = float(validity)
            
            # Overall quality score (weighted average)
            weights = {'completeness': 0.4, 'consistency': 0.3, 'validity': 0.3}
            overall_score = (
                metrics["completeness_score"] * weights['completeness'] +
                metrics["consistency_score"] * weights['consistency'] +
                metrics["validity_score"] * weights['validity']
            )
            metrics["data_quality_score"] = float(overall_score)
            
        except Exception as e:
            logger.warning(f"Error calculating validation metrics: {e}")
        
        return metrics
    
    def _summarize_validation(self, validation_results: Dict):
        """Summarize validation results"""
        checks = validation_results["checks"]
        
        for check in checks:
            validation_results["summary"]["total_checks"] += 1
            
            if check["status"] == "PASS":
                validation_results["summary"]["passed_checks"] += 1
            elif check["status"] == "FAIL":
                validation_results["summary"]["failed_checks"] += 1
            elif check["status"] == "WARNING":
                validation_results["summary"]["warnings"] += 1
        
        # Determine overall status
        if validation_results["summary"]["failed_checks"] > 0:
            validation_results["summary"]["overall_status"] = "FAIL"
        elif validation_results["summary"]["warnings"] > 0:
            validation_results["summary"]["overall_status"] = "WARNING"
        else:
            validation_results["summary"]["overall_status"] = "PASS"
    
    def _generate_recommendations(self, checks: List[Dict]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        for check in checks:
            if check["status"] in ["FAIL", "WARNING"] and check.get("issues"):
                for issue in check["issues"]:
                    # Generate recommendation based on issue
                    if "missing values" in issue.lower():
                        recommendations.append(
                            "Consider imputing missing values or removing records with excessive missing data"
                        )
                    elif "duplicate" in issue.lower():
                        recommendations.append(
                            "Remove duplicate records to avoid bias in model training"
                        )
                    elif "imbalance" in issue.lower():
                        recommendations.append(
                            "Consider using techniques like oversampling, undersampling, or class weights"
                        )
                    elif "outlier" in issue.lower():
                        recommendations.append(
                            "Investigate outliers to determine if they should be removed or transformed"
                        )
                    elif "correlation" in issue.lower():
                        recommendations.append(
                            "Consider removing highly correlated features to reduce multicollinearity"
                        )
                    elif "text quality" in issue.lower():
                        recommendations.append(
                            "Clean text data: remove HTML, normalize text, filter very short/long texts"
                        )
        
        # Remove duplicates
        return list(set(recommendations))
    
    def _save_validation_report(self, validation_results: Dict, dataset_name: str):
        """Save validation report to file"""
        try:
            reports_dir = Path("data/processed/statistics/validation_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON report
            json_path = reports_dir / f"validation_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            # Save summary as markdown
            md_path = reports_dir / f"validation_summary_{dataset_name}.md"
            self._save_validation_summary_markdown(validation_results, md_path)
            
            logger.info(f"Validation report saved to {json_path}")
            
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")
    
    def _save_validation_summary_markdown(self, validation_results: Dict, output_path: Path):
        """Save validation summary as markdown"""
        summary = validation_results["summary"]
        basic_info = validation_results["basic_info"]
        metrics = validation_results.get("metrics", {})
        
        md_content = f"""# Data Validation Report

## Dataset Information
- **Dataset**: {validation_results['dataset_name']}
- **Timestamp**: {validation_results['timestamp']}
- **Rows**: {basic_info['rows']:,}
- **Columns**: {basic_info['columns']}
- **Memory Usage**: {basic_info['memory_mb']:.2f} MB

## Validation Summary
- **Overall Status**: **{summary['overall_status']}**
- **Total Checks**: {summary['total_checks']}
- **Passed Checks**: {summary['passed_checks']}
- **Failed Checks**: {summary['failed_checks']}
- **Warnings**: {summary['warnings']}

## Quality Metrics
- **Data Quality Score**: {metrics.get('data_quality_score', 0):.1f}/100
- **Completeness Score**: {metrics.get('completeness_score', 0):.1f}/100
- **Consistency Score**: {metrics.get('consistency_score', 0):.1f}/100
- **Validity Score**: {metrics.get('validity_score', 0):.1f}/100

## Detailed Checks

| Check | Status | Issues |
|-------|--------|--------|
"""
        
        for check in validation_results["checks"]:
            issues = ", ".join(check.get("issues", []))[:100]  # Limit length
            md_content += f"| {check['check_name']} | {check['status']} | {issues} |\n"
        
        # Add recommendations
        if validation_results["recommendations"]:
            md_content += "\n## Recommendations\n\n"
            for rec in validation_results["recommendations"]:
                md_content += f"- {rec}\n"
        
        # Add issues
        if validation_results["issues"]:
            md_content += "\n## Critical Issues\n\n"
            for issue in validation_results["issues"]:
                md_content += f"- {issue}\n"
        
        with open(output_path, 'w') as f:
            f.write(md_content)
    
    def get_validation_history(self) -> pd.DataFrame:
        """Get validation history as DataFrame"""
        return pd.DataFrame(self.validation_history)
    
    def validate_data_pipeline(
        self,
        raw_data_path: str,
        processed_data_path: str,
        validate_both: bool = True
    ) -> Dict[str, Any]:
        """
        Validate entire data pipeline
        
        Args:
            raw_data_path: Path to raw data
            processed_data_path: Path to processed data
            validate_both: Whether to validate both raw and processed data
        
        Returns:
            Pipeline validation results
        """
        pipeline_results = {
            "pipeline_validation": {},
            "raw_data_validation": None,
            "processed_data_validation": None,
            "consistency_check": {}
        }
        
        try:
            # Load raw data
            raw_df = pd.read_csv(raw_data_path)
            pipeline_results["raw_data_validation"] = self.validate_dataset(
                raw_df, "raw_data", "label", "text"
            )
            
            if validate_both:
                # Load processed data
                processed_df = pd.read_csv(processed_data_path)
                pipeline_results["processed_data_validation"] = self.validate_dataset(
                    processed_df, "processed_data", "label", "text"
                )
                
                # Check consistency between raw and processed
                pipeline_results["consistency_check"] = self._check_data_consistency(
                    raw_df, processed_df
                )
            
            # Overall pipeline status
            pipeline_status = "PASS"
            if pipeline_results["raw_data_validation"]["summary"]["overall_status"] == "FAIL":
                pipeline_status = "FAIL"
            elif validate_both and pipeline_results["processed_data_validation"]["summary"]["overall_status"] == "FAIL":
                pipeline_status = "FAIL"
            
            pipeline_results["pipeline_validation"] = {
                "status": pipeline_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            pipeline_results["pipeline_validation"] = {
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        return pipeline_results
    
    def _check_data_consistency(self, raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict:
        """Check consistency between raw and processed data"""
        consistency_check = {
            "status": "PASS",
            "issues": [],
            "details": {}
        }
        
        try:
            # Check row count consistency
            raw_rows = len(raw_df)
            processed_rows = len(processed_df)
            
            consistency_check["details"]["row_counts"] = {
                "raw": raw_rows,
                "processed": processed_rows,
                "difference": raw_rows - processed_rows,
                "percentage_change": ((processed_rows - raw_rows) / raw_rows * 100) if raw_rows > 0 else 0
            }
            
            if raw_rows != processed_rows:
                consistency_check["status"] = "WARNING"
                consistency_check["issues"].append(
                    f"Row count changed from {raw_rows} to {processed_rows}"
                )
            
            # Check column consistency
            raw_cols = set(raw_df.columns)
            processed_cols = set(processed_df.columns)
            added_cols = processed_cols - raw_cols
            removed_cols = raw_cols - processed_cols
            
            consistency_check["details"]["column_changes"] = {
                "added_columns": list(added_cols),
                "removed_columns": list(removed_cols)
            }
            
            if added_cols or removed_cols:
                consistency_check["status"] = "WARNING"
                if added_cols:
                    consistency_check["issues"].append(f"Added columns: {added_cols}")
                if removed_cols:
                    consistency_check["issues"].append(f"Removed columns: {removed_cols}")
            
            # Check label distribution consistency
            if "label" in raw_df.columns and "label" in processed_df.columns:
                raw_dist = raw_df["label"].value_counts(normalize=True)
                processed_dist = processed_df["label"].value_counts(normalize=True)
                
                # Calculate distribution difference
                diff = {}
                for label in set(raw_dist.index) | set(processed_dist.index):
                    raw_pct = raw_dist.get(label, 0)
                    processed_pct = processed_dist.get(label, 0)
                    diff[label] = abs(raw_pct - processed_pct)
                
                consistency_check["details"]["label_distribution"] = {
                    "raw": raw_dist.to_dict(),
                    "processed": processed_dist.to_dict(),
                    "differences": diff
                }
                
                # Check for significant changes
                significant_changes = {
                    label: diff for label, diff in diff.items() if diff > 0.05
                }
                if significant_changes:
                    consistency_check["status"] = "WARNING"
                    consistency_check["issues"].append(
                        f"Significant label distribution changes: {significant_changes}"
                    )
            
        except Exception as e:
            consistency_check["status"] = "ERROR"
            consistency_check["issues"].append(f"Error in consistency check: {str(e)}")
        
        return consistency_check
