"""
Data validation module for schema, type, range, and distribution checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class DataValidator:
    """
    Comprehensive data validation with schema, type, range, and distribution checks.
    
    Attributes:
        config: Validation configuration
        validation_results: Dictionary storing validation results
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataValidator.
        
        Args:
            config: Validation configuration dictionary
        """
        self.config = config.get('validation', {})
        self.validation_results: Dict[str, Any] = {}
        
        if not self.config.get('enabled', True):
            logger.info("Data validation is disabled in config")
    
    def validate_all(self, df: pd.DataFrame, stage: str = 'raw') -> pd.DataFrame:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            stage: Pipeline stage (raw, processed)
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValidationError: If any validation fails
        """
        if not self.config.get('enabled', True):
            logger.info("Skipping validation (disabled)")
            return df
        
        logger.info("="*60)
        logger.info(f"DATA VALIDATION - {stage.upper()} STAGE")
        logger.info("="*60)
        
        try:
            # Schema validation
            if self.config.get('schema_validation', {}).get('enabled', True):
                self._validate_schema(df)
            
            # Data type validation
            if self.config.get('dtype_validation', {}).get('enabled', True):
                self._validate_dtypes(df)
            
            # Range validation
            if self.config.get('range_validation', {}).get('enabled', True):
                self._validate_ranges(df)
            
            logger.info("✓ All validation checks passed")
            return df
            
        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}", exc_info=True)
            raise ValidationError(f"Validation error: {e}")
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame schema (column names).
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValidationError: If schema validation fails
        """
        logger.info("Validating schema...")
        
        expected_cols = self.config['schema_validation'].get('expected_columns', [])
        actual_cols = df.columns.tolist()
        
        # Check missing columns
        missing_cols = set(expected_cols) - set(actual_cols)
        if missing_cols:
            raise ValidationError(f"Missing columns: {missing_cols}")
        
        # Check extra columns (warning, not error)
        extra_cols = set(actual_cols) - set(expected_cols)
        if extra_cols:
            logger.warning(f"Extra columns found (not in schema): {extra_cols}")
        
        logger.info(f"✓ Schema validation passed: {len(actual_cols)} columns")
    
    def _validate_dtypes(self, df: pd.DataFrame) -> None:
        """
        Validate data types.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValidationError: If dtype validation fails
        """
        logger.info("Validating data types...")
        
        dtype_config = self.config['dtype_validation']
        expected_numeric = dtype_config.get('numeric', [])
        expected_categorical = dtype_config.get('categorical', [])
        
        errors = []
        
        # Check numeric columns
        for col in expected_numeric:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"{col}: expected numeric, got {df[col].dtype}")
        
        # Check categorical columns
        for col in expected_categorical:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype not in ['category']:
                    logger.warning(f"{col}: expected categorical, got {df[col].dtype} (will be encoded)")
        
        if errors:
            raise ValidationError(f"Data type validation failed:\n" + "\n".join(errors))
        
        logger.info(f"✓ Data type validation passed")
    
    def _validate_ranges(self, df: pd.DataFrame) -> None:
        """
        Validate numeric column ranges.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValidationError: If range validation fails
        """
        logger.info("Validating numeric ranges...")
        
        range_checks = self.config['range_validation'].get('checks', {})
        errors = []
        warnings = []
        
        for col, bounds in range_checks.items():
            if col not in df.columns:
                continue
            
            min_val = bounds.get('min')
            max_val = bounds.get('max')
            
            # Check minimum
            if min_val is not None:
                violations = df[df[col] < min_val]
                if len(violations) > 0:
                    errors.append(f"{col}: {len(violations)} values < {min_val} (min: {df[col].min()})")
            
            # Check maximum
            if max_val is not None:
                violations = df[df[col] > max_val]
                if len(violations) > 0:
                    errors.append(f"{col}: {len(violations)} values > {max_val} (max: {df[col].max()})")
        
        if errors:
            raise ValidationError(f"Range validation failed:\n" + "\n".join(errors))
        
        logger.info(f"✓ Range validation passed")
    
    def check_distribution_shift(
        self,
        df_train: pd.DataFrame,
        df_compare: pd.DataFrame,
        numeric_cols: List[str],
        compare_name: str = 'validation'
    ) -> Dict[str, Any]:
        """
        Check for distribution shift between datasets using KS test.
        
        Args:
            df_train: Training DataFrame
            df_compare: Comparison DataFrame (val or test)
            numeric_cols: List of numeric columns to check
            compare_name: Name of comparison set
            
        Returns:
            Dictionary with shift detection results
        """
        if not self.config.get('distribution_shift', {}).get('enabled', True):
            logger.info("Distribution shift detection disabled")
            return {}
        
        logger.info(f"Checking distribution shift: train vs {compare_name}...")
        
        threshold = self.config['distribution_shift'].get('threshold', 0.05)
        method = self.config['distribution_shift'].get('method', 'ks_test')
        
        shift_results = {}
        shifted_features = []
        
        for col in numeric_cols:
            if col not in df_train.columns or col not in df_compare.columns:
                continue
            
            train_data = df_train[col].dropna()
            compare_data = df_compare[col].dropna()
            
            if method == 'ks_test':
                statistic, p_value = stats.ks_2samp(train_data, compare_data)
                
                shift_detected = p_value < threshold
                
                shift_results[col] = {
                    'statistic': round(float(statistic), 4),
                    'p_value': round(float(p_value), 6),
                    'shift_detected': shift_detected,
                    'threshold': threshold
                }
                
                if shift_detected:
                    shifted_features.append(col)
                    logger.warning(f"{col}: Distribution shift detected (p={p_value:.4f})")
                else:
                    logger.info(f"{col}: No significant shift (p={p_value:.4f})")
        
        if shifted_features:
            logger.warning(f"⚠ Distribution shift detected in {len(shifted_features)} features: {shifted_features}")
        else:
            logger.info(f"✓ No significant distribution shifts detected")
        
        return shift_results