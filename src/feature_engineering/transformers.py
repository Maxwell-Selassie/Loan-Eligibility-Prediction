"""
Mathematical transformations for feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.preprocessing import PolynomialFeatures
import logging

logger = logging.getLogger(__name__)


class LogTransformer:
    """
    Apply logarithmic transformations to numeric features.
    
    Attributes:
        config: Log transformation configuration
        columns_to_transform: List of columns to transform
        transformation_stats: Statistics about transformations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LogTransformer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('log_transforms', {})
        self.columns_to_transform = list(self.config.keys())
        self.transformation_stats: Dict[str, Any] = {}
    
    def fit(self, df: pd.DataFrame) -> 'LogTransformer':
        """
        Fit transformer (compute statistics).
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self
        """
        logger.info("Fitting log transformer...")
        
        for col in self.columns_to_transform:
            if col in df.columns:
                self.transformation_stats[col] = {
                    'original_min': float(df[col].min()),
                    'original_max': float(df[col].max()),
                    'original_mean': float(df[col].mean()),
                    'original_skew': float(df[col].skew()),
                    'has_zeros': bool((df[col] == 0).any()),
                    'has_negatives': bool((df[col] < 0).any())
                }
                
                # Validate no negative values for log
                if self.transformation_stats[col]['has_negatives']:
                    logger.error(f"{col}: Contains negative values, cannot apply log transform!")
                    raise ValueError(f"Cannot apply log transform to {col}: negative values present")
                
                logger.info(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, skew={df[col].skew():.2f}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformations.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Applying log transformations...")
        
        df_transformed = df.copy()
        
        for col, col_config in self.config.items():
            if col not in df_transformed.columns:
                logger.warning(f"{col} not found in DataFrame, skipping")
                continue
            
            method = col_config.get('method', 'log1p')
            suffix = col_config.get('suffix', '_log')
            keep_original = col_config.get('keep_original', True)
            
            new_col_name = f"{col}{suffix}"
            
            # Apply transformation
            if method == 'log1p':
                df_transformed[new_col_name] = np.log1p(df_transformed[col])
            elif method == 'log':
                # Add small epsilon to avoid log(0)
                df_transformed[new_col_name] = np.log(df_transformed[col] + 1e-8)
            elif method == 'log10':
                df_transformed[new_col_name] = np.log10(df_transformed[col] + 1e-8)
            else:
                raise ValueError(f"Unknown log method: {method}")
            
            # Check for inf/nan after transformation
            if np.isinf(df_transformed[new_col_name]).any():
                logger.error(f"{new_col_name}: Contains Inf values after transformation!")
                raise ValueError(f"Inf values in {new_col_name}")
            
            if df_transformed[new_col_name].isna().any():
                logger.error(f"{new_col_name}: Contains NaN values after transformation!")
                raise ValueError(f"NaN values in {new_col_name}")
            
            # Calculate skewness after transformation
            new_skew = df_transformed[new_col_name].skew()
            
            logger.info(f"  ✓ {col} → {new_col_name} (skew: {df_transformed[col].skew():.2f} → {new_skew:.2f})")
            
            # Optionally drop original
            if not keep_original:
                df_transformed = df_transformed.drop(columns=[col])
                logger.info(f"    Dropped original column: {col}")
        
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


class PolynomialFeatureGenerator:
    """
    Generate polynomial features (x^2, x*y, etc.).
    
    Attributes:
        config: Polynomial feature configuration
        poly: Sklearn PolynomialFeatures object
        feature_names: Generated feature names
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PolynomialFeatureGenerator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('polynomial_features', {})
        self.poly = None
        self.feature_names: List[str] = []
        self.columns_to_transform: List[str] = []
    
    def fit(self, df: pd.DataFrame) -> 'PolynomialFeatureGenerator':
        """
        Fit polynomial feature generator.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self
        """
        if not self.config.get('enabled', False):
            logger.info("Polynomial features disabled")
            return self
        
        logger.info("Fitting polynomial feature generator...")
        
        degree = self.config.get('degree', 2)
        interaction_only = self.config.get('interaction_only', False)
        include_bias = self.config.get('include_bias', False)
        
        self.columns_to_transform = self.config.get('columns', [])
        
        # Filter columns that exist in dataframe
        self.columns_to_transform = [col for col in self.columns_to_transform if col in df.columns]
        
        if not self.columns_to_transform:
            logger.warning("No columns specified for polynomial features")
            return self
        
        # Initialize PolynomialFeatures
        self.poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        # Fit
        self.poly.fit(df[self.columns_to_transform])
        
        # Get feature names
        self.feature_names = self.poly.get_feature_names_out(self.columns_to_transform)
        
        logger.info(f"  Degree: {degree}")
        logger.info(f"  Input features: {len(self.columns_to_transform)}")
        logger.info(f"  Output features: {len(self.feature_names)}")
        logger.info(f"  Sample features: {list(self.feature_names[:5])}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate polynomial features.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with polynomial features
        """
        if not self.config.get('enabled', False) or self.poly is None:
            return df
        
        logger.info("Generating polynomial features...")
        
        df_transformed = df.copy()
        
        # Generate polynomial features
        poly_features = self.poly.transform(df_transformed[self.columns_to_transform])
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(
            poly_features,
            columns=self.feature_names,
            index=df_transformed.index
        )
        
        # Keep original features or not
        keep_original = self.config.get('keep_original', True)
        
        if keep_original:
            # Drop duplicate columns (original features are included in poly_df)
            original_cols_in_poly = [col for col in self.columns_to_transform if col in self.feature_names]
            poly_df = poly_df.drop(columns=original_cols_in_poly, errors='ignore')
            
            # Concatenate
            df_transformed = pd.concat([df_transformed, poly_df], axis=1)
        else:
            # Replace original columns
            df_transformed = df_transformed.drop(columns=self.columns_to_transform)
            df_transformed = pd.concat([df_transformed, poly_df], axis=1)
        
        logger.info(f"  ✓ Added {len(poly_df.columns)} polynomial features")
        logger.info(f"  Shape: {df.shape} → {df_transformed.shape}")
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            DataFrame with polynomial features
        """
        return self.fit(df).transform(df)