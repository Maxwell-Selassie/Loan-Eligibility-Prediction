"""
Encoding module for categorical and target variables.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class DataEncoder:
    """
    Encode categorical variables with one-hot and label encoding.
    
    Attributes:
        config: Encoding configuration
        one_hot_columns: Columns for one-hot encoding
        label_encode_config: Label encoding configuration
        encoding_mappings: Dictionary storing encoding mappings
        encoded_columns: List of columns after encoding
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataEncoder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('encoding', {})
        self.one_hot_columns = list(self.config.get('one_hot', {}).keys())
        self.label_encode_config = self.config.get('label_encode', {})
        self.encoding_mappings: Dict[str, Any] = {}
        self.encoded_columns: List[str] = []
    
    def fit(self, df: pd.DataFrame) -> 'DataEncoder':
        """
        Fit encoder on training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self
        """
        logger.info("Fitting encoder...")
        
        # Store original columns
        self.original_columns = df.columns.tolist()
        
        # One-hot encoding mappings
        for col in self.one_hot_columns:
            if col in df.columns:
                unique_vals = df[col].unique().tolist()
                self.encoding_mappings[col] = {
                    'type': 'one_hot',
                    'unique_values': unique_vals,
                    'drop_first': self.config['one_hot'][col].get('drop_first', True)
                }
                logger.info(f"  One-hot: {col} → {len(unique_vals)} categories")
        
        # Label encoding mappings
        for col, col_config in self.label_encode_config.items():
            if col in df.columns:
                positive_class = col_config.get('positive_class', 'Y')
                negative_class = col_config.get('negative_class', 'N')
                
                self.encoding_mappings[col] = {
                    'type': 'label',
                    'mapping': {positive_class: 1, negative_class: 0}
                }
                logger.info(f"  Label: {col} → {positive_class}=1, {negative_class}=0")
        
        logger.info(f"✓ Encoder fitted on {len(self.encoding_mappings)} columns")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using fitted encodings.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Encoded DataFrame
        """
        logger.info("Applying encodings...")
        
        df_encoded = df.copy()
        
        # Apply label encoding first
        for col, mapping_info in self.encoding_mappings.items():
            if mapping_info['type'] == 'label' and col in df_encoded.columns:
                mapping = mapping_info['mapping']
                df_encoded[col] = df_encoded[col].map(mapping)
                
                # Check for unmapped values
                if df_encoded[col].isnull().any():
                    unmapped = df[~df[col].isin(mapping.keys())][col].unique()
                    logger.warning(f"{col}: Unmapped values found: {unmapped}")
                
                logger.info(f"  Label encoded: {col}")
        
        # Apply one-hot encoding
        one_hot_cols_present = [col for col in self.one_hot_columns if col in df_encoded.columns]
        
        if one_hot_cols_present:
            df_encoded = pd.get_dummies(
                df_encoded,
                columns=one_hot_cols_present,
                drop_first=True,  # Always drop first to avoid multicollinearity
                dtype=int
            )
            logger.info(f"  One-hot encoded: {len(one_hot_cols_present)} columns")
        
        # Store final column list
        self.encoded_columns = df_encoded.columns.tolist()
        
        logger.info(f"Shape after encoding: {df_encoded.shape}")
        logger.info(f"New columns: {len(df_encoded.columns)} (was {len(df.columns)})")
        
        return df_encoded
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Encoded DataFrame
        """
        return self.fit(df).transform(df)
    
    def save(self, filepath: str) -> None:
        """
        Save encoder to file.
        
        Args:
            filepath: Path to save encoder
        """
        encoder_data = {
            'encoding_mappings': self.encoding_mappings,
            'encoded_columns': self.encoded_columns,
            'original_columns': self.original_columns,
            'config': self.config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(encoder_data, f)
        
        logger.info(f"Encoder saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataEncoder':
        """
        Load encoder from file.
        
        Args:
            filepath: Path to load encoder from
            
        Returns:
            Loaded encoder instance
        """
        with open(filepath, 'rb') as f:
            encoder_data = pickle.load(f)
        
        encoder = cls(config={'encoding': encoder_data['config']})
        encoder.encoding_mappings = encoder_data['encoding_mappings']
        encoder.encoded_columns = encoder_data['encoded_columns']
        encoder.original_columns = encoder_data['original_columns']
        
        logger.info(f"Encoder loaded from: {filepath}")
        
        return encoder

