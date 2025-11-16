"""
Data loading module for model training .
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingDataLoader:
    """
    Load preprocessed data for model training.
    
    Attributes:
        config: Configuration dictionary
        X_train, y_train: Training data
        X_val, y_val: Test data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TrainingDataLoader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('data', {})
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        self.feature_names = None
        self.n_features = None
    
    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load train and test datasets.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        logger.info("="*80)
        logger.info("LOADING TRAINING DATA")
        logger.info("="*80)
        
        target_col = self.config.get('target_column', 'Loan_Status')
        
        # Load train
        train_path = self.config.get('train_data')
        logger.info(f"Loading training data from: {train_path}")
        df_train = pd.read_csv(train_path)
        
        if target_col not in df_train.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data")
        
        self.X_train = df_train.drop(columns=[target_col]).values
        self.y_train = df_train[target_col].values
        self.feature_names = df_train.drop(columns=[target_col]).columns.tolist()
        self.n_features = len(self.feature_names)
        
        logger.info(f"  Train: X={self.X_train.shape}, y={self.y_train.shape}")
        logger.info(f"  Features: {self.n_features}")
        logger.info(f"  Class distribution: {np.bincount(self.y_train.astype(int))}")
        
        # Load test
        test_path = self.config.get('test_data')
        logger.info(f"Loading test data from: {test_path}")
        df_test = pd.read_csv(test_path)
        
        self.X_val = df_test.drop(columns=[target_col]).values
        self.y_val = df_test[target_col].values
        
        logger.info(f"  Test: X={self.X_val.shape}, y={self.y_val.shape}")
        logger.info(f"  Class distribution: {np.bincount(self.y_val.astype(int))}")
        
        # Validate shapes
        if self.X_train.shape[1] != self.X_val.shape[1]:
            raise ValueError("Feature count mismatch between train/test sets")
        
        logger.info("✓ Data loaded successfully")
        
        return self.X_train, self.y_train, self.X_val, self.y_val
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Returns:
            Dictionary with dataset metadata
        """
        return {
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'train_size': len(self.X_train),
            'test_size': len(self.X_val),
            'train_class_dist': np.bincount(self.y_train.astype(int)).tolist(),
            'test_class_dist': np.bincount(self.y_val.astype(int)).tolist()
        }