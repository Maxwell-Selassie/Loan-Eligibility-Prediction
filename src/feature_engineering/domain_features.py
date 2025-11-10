"""
Domain-specific features for loan eligibility prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DomainFeatureCreator:
    """
    Create domain-specific features based on lending industry knowledge.
    
    Attributes:
        config: Domain feature configuration
        feature_formulas: Dictionary mapping feature names to formulas
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DomainFeatureCreator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('domain_features', {})
        self.feature_formulas: Dict[str, Dict[str, Any]] = {}
    
    def fit(self, df: pd.DataFrame) -> 'DomainFeatureCreator':
        """
        Fit domain feature creator.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self
        """
        logger.info("Fitting domain feature creator...")
        
        for feature_name, feature_config in self.config.items():
            formula = feature_config.get('formula')
            dtype = feature_config.get('dtype', 'float')
            self.feature_formulas[feature_name] = {
                'formula': formula,
                'dtype': dtype
            }
            reason = feature_config.get('reason', 'Not specified')
            logger.info(f"  {feature_name}: {formula}")
            logger.info(f"    Reason: {reason}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with domain features
        """
        logger.info("Creating domain-specific features...")
        
        df_transformed = df.copy()
        
        for feature_name, feature_info in self.feature_formulas.items():
            formula = feature_info['formula']
            dtype = feature_info['dtype']
            
            try:
                # Safe evaluation
                safe_dict = {
                    'df': df_transformed,
                    'np': np,
                    'abs': abs,
                    'max': max,
                    'min': min
                }
                
                # Evaluate formula
                df_transformed[feature_name] = eval(formula, {"__builtins__": {}}, safe_dict)
                
                # Handle inf/nan
                if np.isinf(df_transformed[feature_name]).any():
                    n_inf = np.isinf(df_transformed[feature_name]).sum()
                    logger.warning(f"{feature_name}: Contains {n_inf} Inf values, replacing with 0")
                    df_transformed[feature_name] = df_transformed[feature_name].replace([np.inf, -np.inf], 0)
                
                if df_transformed[feature_name].isna().any():
                    n_nan = df_transformed[feature_name].isna().sum()
                    logger.warning(f"{feature_name}: Contains {n_nan} NaN values, filling with 0")
                    df_transformed[feature_name] = df_transformed[feature_name].fillna(0)
                
                # Cast to specified dtype
                if dtype == 'int':
                    df_transformed[feature_name] = df_transformed[feature_name].astype(int)
                elif dtype == 'float':
                    df_transformed[feature_name] = df_transformed[feature_name].astype(float)
                
                logger.info(f"  ✓ {feature_name}: mean={df_transformed[feature_name].mean():.4f}")
                
            except Exception as e:
                logger.error(f"Failed to create feature {feature_name}: {e}")
                raise
        
        logger.info(f"  Created {len(self.feature_formulas)} domain features")
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            DataFrame with domain features
        """
        return self.fit(df).transform(df)