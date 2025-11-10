"""
Interaction feature creation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class InteractionFeatureCreator:
    """
    Create interaction features from existing columns.
    
    Attributes:
        config: Interaction feature configuration
        feature_formulas: Dictionary mapping feature names to formulas
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize InteractionFeatureCreator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('interaction_features', {})
        self.feature_formulas: Dict[str, str] = {}
    
    def fit(self, df: pd.DataFrame) -> 'InteractionFeatureCreator':
        """
        Fit interaction creator (store formulas).
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self
        """
        logger.info("Fitting interaction feature creator...")
        
        for feature_name, feature_config in self.config.items():
            formula = feature_config.get('formula')
            self.feature_formulas[feature_name] = formula
            reason = feature_config.get('reason', 'Not specified')
            logger.info(f"  {feature_name}: {formula}")
            logger.info(f"    Reason: {reason}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        df_transformed = df.copy()
        
        for feature_name, formula in self.feature_formulas.items():
            try:
                # Safely evaluate formula
                # Create a safe namespace with only necessary functions
                safe_dict = {
                    'df': df_transformed,
                    'np': np,
                    'abs': abs,
                    'max': max,
                    'min': min
                }
                
                # Evaluate formula
                df_transformed[feature_name] = eval(formula, {"__builtins__": {}}, safe_dict)
                
                # Check for inf/nan
                if np.isinf(df_transformed[feature_name]).any():
                    n_inf = np.isinf(df_transformed[feature_name]).sum()
                    logger.warning(f"{feature_name}: Contains {n_inf} Inf values, replacing with NaN")
                    df_transformed[feature_name] = df_transformed[feature_name].replace([np.inf, -np.inf], np.nan)
                
                if df_transformed[feature_name].isna().any():
                    n_nan = df_transformed[feature_name].isna().sum()
                    logger.warning(f"{feature_name}: Contains {n_nan} NaN values, filling with 0")
                    df_transformed[feature_name] = df_transformed[feature_name].fillna(0)
                
                logger.info(f"  ✓ {feature_name}: mean={df_transformed[feature_name].mean():.2f}, std={df_transformed[feature_name].std():.2f}")
                
            except Exception as e:
                logger.error(f"Failed to create feature {feature_name}: {e}")
                raise
        
        logger.info(f"  Created {len(self.feature_formulas)} interaction features")
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            DataFrame with interaction features
        """
        return self.fit(df).transform(df)