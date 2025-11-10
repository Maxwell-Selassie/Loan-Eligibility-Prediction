"""
Feature validation for engineered features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class FeatureValidator:
    """
    Validate engineered features for common issues.
    
    Attributes:
        config: Validation configuration
        validation_results: Dictionary storing validation results
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureValidator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('validation', {})
        self.validation_results: Dict[str, Any] = {}
    
    def validate(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """
        Validate engineered features.
        
        Args:
            df: DataFrame with engineered features
            feature_names: List of newly created feature names
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If critical validation fails
        """
        if not self.config.get('enabled', True):
            logger.info("Feature validation disabled")
            return df
        
        logger.info("="*60)
        logger.info("FEATURE VALIDATION")
        logger.info("="*60)
        
        df_validated = df.copy()
        issues_found = []
        
        # 1. Check for Inf values
        if self.config.get('check_inf', True):
            for col in feature_names:
                if col in df_validated.columns:
                    n_inf = np.isinf(df_validated[col]).sum()
                    if n_inf > 0:
                        issues_found.append(f"{col}: {n_inf} Inf values")
                        logger.error(f"  ✗ {col}: {n_inf} Inf values detected!")
        
        # 2. Check for NaN values
        if self.config.get('check_nan', True):
            for col in feature_names:
                if col in df_validated.columns:
                    n_nan = df_validated[col].isna().sum()
                    if n_nan > 0:
                        issues_found.append(f"{col}: {n_nan} NaN values")
                        logger.error(f"  ✗ {col}: {n_nan} NaN values detected!")
        
        # 3. Check for negative values after log transform
        if self.config.get('check_negative_after_log', True):
            log_columns = [col for col in feature_names if '_log' in col]
            for col in log_columns:
                if col in df_validated.columns:
                    n_negative = (df_validated[col] < 0).sum()
                    if n_negative > 0:
                        issues_found.append(f"{col}: {n_negative} negative values")
                        logger.error(f"  ✗ {col}: {n_negative} negative values after log transform!")
        
        # 4. Check skewness
        if self.config.get('check_skewness', False):
            skewness_threshold = self.config.get('skewness_threshold', 1.0)
            for col in feature_names:
                if col in df_validated.columns and pd.api.types.is_numeric_dtype(df_validated[col]):
                    skew = df_validated[col].skew()
                    if abs(skew) > skewness_threshold:
                        logger.warning(f"  ⚠ {col}: High skewness ({skew:.2f})")
        
        # 5. Check for constant features (zero variance)
        for col in feature_names:
            if col in df_validated.columns:
                if df_validated[col].nunique() == 1:
                    issues_found.append(f"{col}: Constant feature (zero variance)")
                    logger.warning(f"  ⚠ {col}: Constant feature detected")
        
        # 6. Log summary statistics
        logger.info("\nFeature Statistics:")
        for col in feature_names[:10]:  # Log first 10 features
            if col in df_validated.columns and pd.api.types.is_numeric_dtype(df_validated[col]):
                logger.info(f"  {col}:")
                logger.info(f"    Mean: {df_validated[col].mean():.4f}")
                logger.info(f"    Std: {df_validated[col].std():.4f}")
                logger.info(f"    Min: {df_validated[col].min():.4f}")
                logger.info(f"    Max: {df_validated[col].max():.4f}")
        
        # Fail if critical issues found
        if issues_found:
            logger.error("="*60)
            logger.error("VALIDATION FAILED - CRITICAL ISSUES FOUND:")
            for issue in issues_found:
                logger.error(f"  • {issue}")
            logger.error("="*60)
            raise ValueError(f"Feature validation failed: {len(issues_found)} issues found")
        
        logger.info("✓ All feature validation checks passed")
        
        return df_validated