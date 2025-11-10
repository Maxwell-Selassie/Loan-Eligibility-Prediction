"""
Outlier handling module for future use.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)


class OutlierHandler:
    """
    Handle outliers with various strategies (clip, remove, transform).
    
    Attributes:
        config: Outlier configuration
        outlier_bounds: Dictionary storing outlier bounds per column
        outlier_stats: Statistics about outliers
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OutlierHandler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('outliers', {})
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.outlier_stats: Dict[str, Any] = {}
    
    def fit(self, df: pd.DataFrame, numeric_cols: List[str]) -> 'OutlierHandler':
        """
        Fit outlier detector on training data.
        
        Args:
            df: Training DataFrame
            numeric_cols: List of numeric columns to check
            
        Returns:
            Self
        """
        if not self.config.get('enabled', False):
            logger.info("Outlier handling is disabled")
            return self
        
        logger.info("Fitting outlier detector...")
        
        method = self.config.get('method', 'IQR').upper()
        
        if method == 'IQR':
            self._fit_iqr(df, numeric_cols)
        elif method == 'ZSCORE':
            self._fit_zscore(df, numeric_cols)
        elif method == 'ISOLATION_FOREST':
            self._fit_isolation_forest(df, numeric_cols)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return self
    
    def _fit_iqr(self, df: pd.DataFrame, numeric_cols: List[str]) -> None:
        """Fit IQR-based outlier detection."""
        multiplier = self.config.get('iqr_multiplier', 1.5)
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            self.outlier_bounds[col] = (lower_bound, upper_bound)
            
            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(df)) * 100
            
            self.outlier_stats[col] = {
                'method': 'IQR',
                'count': int(outliers),
                'percentage': round(outlier_pct, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2)
            }
            
            logger.info(f"  {col}: {outliers} outliers ({outlier_pct:.1f}%) | Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    def _fit_zscore(self, df: pd.DataFrame, numeric_cols: List[str]) -> None:
        """Fit Z-score based outlier detection."""
        threshold = 3  # Standard threshold for z-score
        
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            self.outlier_bounds[col] = (lower_bound, upper_bound)
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(df)) * 100
            
            self.outlier_stats[col] = {
                'method': 'Z-Score',
                'count': int(outliers),
                'percentage': round(outlier_pct, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2)
            }
            
            logger.info(f"  {col}: {outliers} outliers ({outlier_pct:.1f}%)")
    
    def _fit_isolation_forest(self, df: pd.DataFrame, numeric_cols: List[str]) -> None:
        """Fit Isolation Forest for outlier detection."""
        contamination = 0.1  # Expected proportion of outliers
        
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(df[numeric_cols])
        
        outlier_mask = predictions == -1
        n_outliers = outlier_mask.sum()
        outlier_pct = (n_outliers / len(df)) * 100
        
        self.outlier_stats['isolation_forest'] = {
            'method': 'Isolation Forest',
            'count': int(n_outliers),
            'percentage': round(outlier_pct, 2)
        }
        
        logger.info(f"  Isolation Forest: {n_outliers} outliers ({outlier_pct:.1f}%)")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using configured strategy.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.config.get('enabled', False):
            return df
        
        if not self.outlier_bounds:
            return df
        
        logger.info("Handling outliers...")
        
        strategy = self.config.get('strategy', 'clip')
        df_handled = df.copy()
        
        if strategy == 'clip':
            for col, (lower, upper) in self.outlier_bounds.items():
                if col in df_handled.columns:
                    df_handled[col] = df_handled[col].clip(lower=lower, upper=upper)
            logger.info(f"  Clipped outliers in {len(self.outlier_bounds)} columns")
        
        elif strategy == 'remove':
            # Remove rows with outliers
            mask = pd.Series([True] * len(df_handled))
            for col, (lower, upper) in self.outlier_bounds.items():
                if col in df_handled.columns:
                    mask &= (df_handled[col] >= lower) & (df_handled[col] <= upper)
            
            rows_before = len(df_handled)
            df_handled = df_handled[mask]
            rows_removed = rows_before - len(df_handled)
            logger.info(f"  Removed {rows_removed} rows with outliers")
        
        elif strategy == 'transform':
            # Log transform for positive skewed data
            for col in self.outlier_bounds.keys():
                if col in df_handled.columns:
                    if (df_handled[col] > 0).all():
                        df_handled[col] = np.log1p(df_handled[col])
            logger.info(f"  Log-transformed {len(self.outlier_bounds)} columns")
        
        return df_handled
    
    def fit_transform(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            numeric_cols: List of numeric columns
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(df, numeric_cols).transform(df)