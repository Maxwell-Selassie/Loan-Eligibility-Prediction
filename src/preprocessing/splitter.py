"""
Data splitting module for train/validation/test splits.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Split data into train, validation, and test sets with stratification.
    
    Attributes:
        config: Split configuration
        split_indices: Dictionary storing split indices
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataSplitter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('data_split', {})
        self.split_indices: Dict[str, np.ndarray] = {}
    
    def split(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            target_col: Target column name for stratification
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("="*60)
        logger.info("SPLITTING DATA")
        logger.info("="*60)
        
        train_size = self.config.get('train_size', 0.80)
        test_size = self.config.get('test_size', 0.20)
        
        # Validate sizes
        total_size = train_size  + test_size
        if not np.isclose(total_size, 1.0):
            raise ValueError(f"Split sizes must sum to 1.0, got {total_size}")
        
        stratify = self.config.get('stratify', True)
        random_state = self.config.get('random_state', 42)
        shuffle = self.config.get('shuffle', True)
        
        logger.info(f"Split configuration:")
        logger.info(f"  Train: {train_size*100:.0f}%")
        logger.info(f"  Test: {test_size*100:.0f}%")
        logger.info(f"  Stratify: {stratify}")
        logger.info(f"  Random state: {random_state}")
        
        # Check target column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # First split: train vs test
        train_val_size = train_size
        
        stratify_col = df[target_col] if stratify else None
        
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_col
        )
        
        
        # Store indices
        self.split_indices = {
            'train': df_train.index.to_numpy(),
            'test': df_test.index.to_numpy()
        }
        
        # Log split results
        logger.info(f"\nSplit results:")
        logger.info(f"  Train: {len(df_train):,} rows ({len(df_train)/len(df)*100:.1f}%)")
        logger.info(f"  Test: {len(df_test):,} rows ({len(df_test)/len(df)*100:.1f}%)")
        logger.info(f"  Total: {len(df):,} rows")
        
        # Log class distribution if stratified
        if stratify:
            logger.info(f"\nClass distribution in '{target_col}':")
            logger.info(f"  Original:")
            for val, count in df[target_col].value_counts().items():
                pct = count / len(df) * 100
                logger.info(f"    {val}: {count:,} ({pct:.1f}%)")
            
            logger.info(f"  Train:")
            for val, count in df_train[target_col].value_counts().items():
                pct = count / len(df_train) * 100
                logger.info(f"    {val}: {count:,} ({pct:.1f}%)")
            
            
            logger.info(f"  Test:")
            for val, count in df_test[target_col].value_counts().items():
                pct = count / len(df_test) * 100
                logger.info(f"    {val}: {count:,} ({pct:.1f}%)")
        
        logger.info("✓ Data split completed successfully")
        
        return df_train, df_test