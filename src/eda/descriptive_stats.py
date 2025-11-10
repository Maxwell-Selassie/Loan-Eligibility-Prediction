"""
Descriptive statistics computation with optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from utils.io_utils import read_yaml

logger = logging.getLogger(__name__)

config = read_yaml('config/EDA_config.yaml')
def compute_basic_stats(df: pd.DataFrame) -> Dict[str, any]:
    """
    Compute basic statistics efficiently.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with basic statistics
    """
    logger.info("Computing basic descriptive statistics...")
    
    stats = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'columns': df.columns.tolist()
    }
    
    logger.info(f"Dataset: {stats['n_rows']:,} rows x {stats['n_columns']} columns")
    logger.info(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
    
    return stats


def analyze_numeric_columns(
    df: pd.DataFrame,
    id_column: Optional[str] = config['data'].get('id_column', None)
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Analyze numeric columns with summary statistics.
    
    Args:
        df: Input DataFrame
        id_column: ID column to exclude from analysis
        
    Returns:
        Tuple of (summary DataFrame, list of numeric columns)
    """
    logger.info("Analyzing numeric columns...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude ID column
    if id_column  in numeric_cols:
        numeric_cols.remove(id_column)
    
    if not numeric_cols:
        logger.warning("No numeric columns found")
        return pd.DataFrame(), []
    
    # Compute statistics (vectorized)
    summary = df[numeric_cols].describe().T
    summary['range'] = summary['max'] - summary['min']
    summary['cv'] = (summary['std'] / summary['mean']).round(4)  # Coefficient of variation
    
    logger.info(f"Analyzed {len(numeric_cols)} numeric columns")
    
    return summary, numeric_cols


def analyze_categorical_columns(
    df: pd.DataFrame,
    target_column: Optional[str] = config['data'].get('target_column', None)
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Analyze categorical columns.
    
    Args:
        df: Input DataFrame
        target_column: Target column (will be included in analysis)
        
    Returns:
        Tuple of (summary DataFrame, list of categorical columns)
    """
    logger.info("Analyzing categorical columns...")
    
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if not cat_cols:
        logger.warning("No categorical columns found")
        return pd.DataFrame(), []
    
    summary_data = []
    for col in cat_cols:
        summary_data.append({
            'column': col,
            'unique_count': df[col].nunique(),
            'most_frequent': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
            'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
            'top_5_values': df[col].value_counts().head(5).to_dict()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    logger.info(f"Analyzed {len(cat_cols)} categorical columns")
    
    return summary_df, cat_cols