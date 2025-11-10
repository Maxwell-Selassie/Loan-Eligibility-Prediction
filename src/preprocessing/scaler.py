"""
Scaling module for numeric feature standardization/normalization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class DataScaler:
    """
    Scale numeric features using various scaling methods.
    
    Attributes:
        config: Scaling configuration
        scaler: Sklearn scaler object
        columns_to_scale: List of columns to scale
        fitted: Whether scaler is fitted
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataScaler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('scaling', {})
        self.columns_to_scale = self.config.get('columns_to_scale', [])
        self.fitted = False
        self.scaler: Optional[Any] = None
        self._initialize_scaler()
    
    def _initialize_scaler(self) -> None:
        """Initialize sklearn scaler based on config."""
        if not self.config.get('enabled', True):
            logger.info("Scaling is disabled in config")
            return
        
        method = self.config.get('method', 'standard').lower()
        
        scaler_map = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
            'maxabs': MaxAbsScaler
        }
        
        if method not in scaler_map:
            raise ValueError(f"Unknown scaling method: {method}. Choose from {list(scaler_map.keys())}")
        
        self.scaler = scaler_map[method]()
        logger.info(f"Initialized {method} scaler")
    
    def fit(self, df: pd.DataFrame) -> 'DataScaler':
        """
        Fit scaler on training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self
        """
        if not self.config.get('enabled', True):
            logger.info("Skipping scaler fitting (disabled)")
            return self
        
        logger.info("Fitting scaler...")
        
        # Filter columns that exist in dataframe
        cols_to_scale = [col for col in self.columns_to_scale if col in df.columns]
        
        if not cols_to_scale:
            logger.warning("No columns to scale found in DataFrame")
            return self
        
        # Fit scaler
        self.scaler.fit(df[cols_to_scale])
        self.fitted = True
        
        # Log scaling parameters
        if hasattr(self.scaler, 'mean_'):
            logger.info("Scaling parameters (mean, std):")
            for col, mean, std in zip(cols_to_scale, self.scaler.mean_, self.scaler.scale_):
                logger.info(f"  {col}: μ={mean:.2f}, σ={std:.2f}")
        elif hasattr(self.scaler, 'min_'):
            logger.info("Scaling parameters (min, max):")
            for col, min_val, max_val in zip(cols_to_scale, self.scaler.min_, self.scaler.scale_):
                logger.info(f"  {col}: min={min_val:.2f}, range={max_val:.2f}")
        
        logger.info(f"✓ Scaler fitted on {len(cols_to_scale)} columns")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using fitted scaler.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Scaled DataFrame
        """
        if not self.config.get('enabled', True):
            logger.info("Skipping scaling (disabled)")
            return df
        
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        logger.info("Applying scaling...")
        
        df_scaled = df.copy()
        
        # Filter columns that exist
        cols_to_scale = [col for col in self.columns_to_scale if col in df_scaled.columns]
        
        if not cols_to_scale:
            logger.warning("No columns to scale found")
            return df_scaled
        
        # Transform
        scaled_values = self.scaler.transform(df_scaled[cols_to_scale])
        df_scaled[cols_to_scale] = scaled_values
        
        logger.info(f"✓ Scaled {len(cols_to_scale)} columns")
        
        # Log sample statistics after scaling
        for col in cols_to_scale[:3]:  # Log first 3 columns
            logger.info(f"  {col}: mean={df_scaled[col].mean():.4f}, std={df_scaled[col].std():.4f}")
        
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Scaled DataFrame
        """
        return self.fit(df).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            df: Scaled DataFrame
            
        Returns:
            DataFrame in original scale
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted")
        
        df_original = df.copy()
        cols_to_scale = [col for col in self.columns_to_scale if col in df_original.columns]
        
        if cols_to_scale:
            original_values = self.scaler.inverse_transform(df_original[cols_to_scale])
            df_original[cols_to_scale] = original_values
        
        return df_original
    
    def save(self, filepath: str) -> None:
        """
        Save scaler to file.
        
        Args:
            filepath: Path to save scaler
        """
        if not self.fitted:
            logger.warning("Saving unfitted scaler")
        
        scaler_data = {
            'scaler': self.scaler,
            'columns_to_scale': self.columns_to_scale,
            'fitted': self.fitted,
            'config': self.config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        logger.info(f"Scaler saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataScaler':
        """
        Load scaler from file.
        
        Args:
            filepath: Path to load scaler from
            
        Returns:
            Loaded scaler instance
        """
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        
        scaler_obj = cls(config={'scaling': scaler_data['config']})
        scaler_obj.scaler = scaler_data['scaler']
        scaler_obj.columns_to_scale = scaler_data['columns_to_scale']
        scaler_obj.fitted = scaler_data['fitted']
        
        logger.info(f"Scaler loaded from: {filepath}")
        
        return scaler_obj