"""
Feature Engineering Pipeline - Main Orchestrator
Integrates all feature engineering modules.
"""

import pandas as pd
from typing import Dict, Any, List
import logging
from pathlib import Path

from src.feature_engineering import (
    LogTransformer,
    PolynomialFeatureGenerator,
    InteractionFeatureCreator,
    DomainFeatureCreator,
    FeatureValidator
)

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline.
    
    Attributes:
        config: Configuration dictionary
        log_transformer: Log transformation module
        interaction_creator: Interaction feature module
        polynomial_generator: Polynomial feature module
        domain_creator: Domain feature module
        validator: Feature validator
        engineered_features: List of newly created features
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Feature Engineering Pipeline.
        
        Args:
            config: Configuration dictionary (from preprocessing config)
        """
        self.config = config.get('feature_engineering', {})
        
        if not self.config.get('enabled', True):
            logger.info("Feature engineering is disabled in config")
            return
        
        # Initialize components
        self.log_transformer = LogTransformer(self.config)
        self.interaction_creator = InteractionFeatureCreator(self.config)
        self.polynomial_generator = PolynomialFeatureGenerator(self.config)
        self.domain_creator = DomainFeatureCreator(self.config)
        self.validator = FeatureValidator(self.config)
        
        self.engineered_features: List[str] = []
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineeringPipeline':
        """
        Fit all feature engineering components on training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self
        """
        if not self.config.get('enabled', True):
            return self
        
        logger.info("="*80)
        logger.info("FITTING FEATURE ENGINEERING PIPELINE")
        logger.info("="*80)
        
        original_columns = df.columns.tolist()
        
        # Fit all components
        self.log_transformer.fit(df)
        self.interaction_creator.fit(df)
        self.polynomial_generator.fit(df)
        self.domain_creator.fit(df)
        
        logger.info("✓ Feature engineering pipeline fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame with engineered features
        """
        if not self.config.get('enabled', True):
            logger.info("Feature engineering disabled, returning original DataFrame")
            return df
        
        logger.info("="*80)
        logger.info("APPLYING FEATURE ENGINEERING TRANSFORMATIONS")
        logger.info("="*80)
        
        original_columns = df.columns.tolist()
        df_transformed = df.copy()
        
        # Stage 1: Log transformations
        logger.info("\n[1/4] Log Transformations")
        logger.info("-"*60)
        df_transformed = self.log_transformer.transform(df_transformed)
        
        # Stage 2: Interaction features
        logger.info("\n[2/4] Interaction Features")
        logger.info("-"*60)
        df_transformed = self.interaction_creator.transform(df_transformed)
        
        # Stage 3: Polynomial features
        logger.info("\n[3/4] Polynomial Features")
        logger.info("-"*60)
        df_transformed = self.polynomial_generator.transform(df_transformed)
        
        # Stage 4: Domain-specific features
        logger.info("\n[4/4] Domain-Specific Features")
        logger.info("-"*60)
        df_transformed = self.domain_creator.transform(df_transformed)
        
        # Track engineered features
        self.engineered_features = [col for col in df_transformed.columns if col not in original_columns]
        
        # Validation
        logger.info("\n[Validation] Checking Engineered Features")
        logger.info("-"*60)
        df_transformed = self.validator.validate(df_transformed, self.engineered_features)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("="*80)
        logger.info(f"Original features: {len(original_columns)}")
        logger.info(f"Engineered features: {len(self.engineered_features)}")
        logger.info(f"Total features: {len(df_transformed.columns)}")
        logger.info(f"Shape: {df.shape} → {df_transformed.shape}")
        
        logger.info(f"\nEngineered feature list:")
        for i, feat in enumerate(self.engineered_features, 1):
            logger.info(f"  {i}. {feat}")
        
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
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of engineered feature names.
        
        Returns:
            List of feature names
        """
        return self.engineered_features