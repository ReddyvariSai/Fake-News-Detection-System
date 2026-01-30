import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation and quality assurance"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.validation_results = {}
    
    def _get_default_config(self) -> Dict:
        """Get default validation configuration"""
        return {
            "quality_thresholds": {
                "missing_values_percentage": 30.0,
                "class_imbalance_ratio": 0.3,
                "min_samples_per_class": 100,
                "text_length_min": 10,
                "text_length_max": 10000
            },
            "validation_rules": {
                "check_missing_values": True,
                "check_class_distribution": True,
                "check_text_quality": True,
                "check_data_types": True,
                "check_duplicates": True,
                "check_outliers": True
            }
        }
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        label_column: Optional[str] = None,
        text_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive dataset validation
        
        Args:
            df: Input DataFrame
            label_column: Name of label column
            text_column: Name of text column
        
        Returns:
            Validation results
        """
        logger.info("Starting dataset validation...")
        
        validation_results = {
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "warnings": 0
            },
            "checks": [],
            "issues": [],
            "recommendations": []
        }
        
        # Check 1: Data structure
        validation_results["checks"].append(
            self._check_data_structure(df)
        )
        
        # Check 2: Missing values
        if self.config["validation_rules"]["check_missing_values"]:
            validation_results["checks"].append(
                self._check_missing_values(df)
            )
        
        # Check 3: Data types
        if self.config["validation_rules"]["check_data_types"]:
            validation_results["checks"].append(
                self._check_data_types(df)
            )
        
        # Check 4: Duplicates
        if self.config["validation_rules"]["check_duplicates"]:
            validation_results["checks"].append(
                self._check_duplicates(df, text_column)
            )
        
        # Check 5: Class distribution (if label column provided)
        if (self.config["validation_rules"]["check_class_distribution"] and 
            label_column and label_column in df.columns):
            validation_results["checks"].append(
                self._check_class_distribution(df, label_column)
            )
        
        # Check 6: Text quality (if text column provided)
        if (self.config["validation_rules"]["check_text_quality"] and 
            text_column and text_column in df.columns):
            validation_results["checks"].append(
                self._check_text_quality(df, text_column)
            )
        
        # Check 7: Outliers (if numeric columns exist)
        if self.config["validation_rules"]["check_outliers"]:
            validation_results["checks"].append(
                self._check_outliers(df)
            )
        
        # Summarize results
        for check in validation_results["checks"]:
            validation_results["summary"]["total_checks"] += 1
            if check["status"] == "PASS":
                validation
