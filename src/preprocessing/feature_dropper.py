"""
Feature dropping module for removing unnecessary columns.
"""

import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class FeatureDropper:
    """
    Drop unnecessary columns based on configuration.
    
    Attributes:
        config: Drop configuration
        dropped_columns: List of dropped column names
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureDropper.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('columns_to_drop', [])
        self.dropped_columns: List[str] = []
    
    def fit(self, df: pd.DataFrame) -> 'FeatureDropper':
        """
        Fit dropper (identify columns to drop).
        
        Args:
            df: DataFrame to fit on
            
        Returns:
            Self
        """
        logger.info("Identifying columns to drop...")
        
        self.dropped_columns = []
        for item in self.config:
            col = item['column']
            if col in df.columns:
                self.dropped_columns.append(col)
                reason = item.get('reason', 'Not specified')
                logger.info(f"  - {col}: {reason}")
        
        if not self.dropped_columns:
            logger.info("No columns to drop")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns from DataFrame.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.dropped_columns:
            return df
        
        logger.info(f"Dropping {len(self.dropped_columns)} columns: {self.dropped_columns}")
        
        df_transformed = df.drop(columns=self.dropped_columns, errors='ignore')
        
        logger.info(f"Shape after dropping: {df_transformed.shape}")
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)