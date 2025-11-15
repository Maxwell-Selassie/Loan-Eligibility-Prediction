"""
Data quality checks: missing values, duplicates, outliers.
Optimized for performance and comprehensive reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from src.utils.io_utils import read_yaml

logger = logging.getLogger(__name__)

config = read_yaml('config/EDA_config.yaml')


def check_missing_values(
    df: pd.DataFrame,
    warning_threshold: float = config['data_quality'].get('missing_threshold_warning', 0.05),
    critical_threshold: float = config['data_quality'].get('missing_threshold_critical', 0.30)
) -> Dict[str, any]:
    """
    Analyze missing values with thresholds.
    
    Args:
        df: Input DataFrame
        warning_threshold: Threshold for warning level
        critical_threshold: Threshold for critical level
        
    Returns:
        Dictionary with missing value analysis
    """
    logger.info("Analyzing missing values...")
    
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        logger.info("No missing values found")
        return {"has_missing": False, "missing_df": pd.DataFrame()}
    
    total_rows = len(df)
    missing_pct = (missing / total_rows).round(4)
    
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_percentage': missing_pct,
        'severity': ['CRITICAL' if pct >= critical_threshold 
                    else 'WARNING' if pct >= warning_threshold 
                    else 'INFO' 
                    for pct in missing_pct]
    })
    
    # Log summary
    critical_cols = missing_df[missing_df['severity'] == 'CRITICAL']
    warning_cols = missing_df[missing_df['severity'] == 'WARNING']
    
    logger.info(f"Missing values found in {len(missing_df)} columns")
    if len(critical_cols) > 0:
        logger.warning(f"CRITICAL: {len(critical_cols)} columns exceed {critical_threshold*100}% missing")
    if len(warning_cols) > 0:
        logger.warning(f"WARNING: {len(warning_cols)} columns exceed {warning_threshold*100}% missing")
    
    return {
        "has_missing": True,
        "missing_df": missing_df,
        "critical_columns": critical_cols.index.tolist(),
        "warning_columns": warning_cols.index.tolist()
    }


def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Check for duplicate rows.
    
    Args:
        df: Input DataFrame
        subset: Columns to check for duplicates
        
    Returns:
        Dictionary with duplicate analysis
    """
    if config['data_quality']['check_duplicates']:
        logger.info("Checking for duplicate rows...")
        
        duplicates = df.duplicated(subset=subset, keep='first')
        n_duplicates = duplicates.sum()
        
        if n_duplicates == 0:
            logger.info("No duplicate rows found")
            return {"has_duplicates": False, "count": 0, "percentage": 0.0}
        
        duplicate_pct = (n_duplicates / len(df)) * 100
        logger.warning(f"Found {n_duplicates:,} duplicate rows ({duplicate_pct:.2f}%)")
        
        return {
            "has_duplicates": True,
            "count": n_duplicates,
            "percentage": duplicate_pct,
            "duplicate_rows": df[duplicates]
        }
    else:
        logger.warning(f'Duplicates checking disabled! (Skipping...)')
        return {}


def detect_outliers_iqr(
    df: pd.DataFrame,
    numeric_columns: List[str],
    multiplier: float = config['statistics'].get('outlier_iqr_multiplier', 1.5),
    id_column: Optional[str] = config['data'].get('id_column', None)
) -> Dict[str, any]:
    """
    Detect outliers using IQR method (vectorized for speed).
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns to check
        multiplier: IQR multiplier (default 1.5)
        id_column: ID column to exclude
        
    Returns:
        Dictionary with outlier analysis per column
    """
    if config['data_quality']['detect_outliers'] and config['data_quality']['outlier_report']:
        logger.info(f"Detecting outliers using IQR method (multiplier={multiplier})...")
        
        # Filter out ID column
        cols_to_check = [col for col in numeric_columns if col != id_column]
        
        outlier_summary = {}
        
        # Vectorized computation
        for col in cols_to_check:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Vectorized outlier detection
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            n_outliers = outlier_mask.sum()
            outlier_pct = (n_outliers / len(df)) * 100
            
            outlier_summary[col] = {
                'count': int(n_outliers),
                'percentage': round(outlier_pct, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'Q1': round(Q1, 2),
                'Q3': round(Q3, 2),
                'IQR': round(IQR, 2)
            }
            
            logger.info(f"{col}: {n_outliers} outliers ({outlier_pct:.2f}%) | Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            logger.info(f'Total Number of outliers in the dataset : {outlier_mask.sum().sum()}')
        
        return outlier_summary
    else:
        logger.warning(f'Outlier detection disabled! (Skipping...)')
        return {}