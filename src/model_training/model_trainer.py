"""
Core model training module with sklearn models.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging
import time

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train classification models with cross-validation.
    
    Attributes:
        config: Configuration dictionary
        model: Trained model instance
        model_name: Name of the model
        training_time: Time taken to train
        cv_scores: Cross-validation scores
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.model_name = None
        self.training_time = 0.0
        self.cv_scores = None
    
    def _get_model_instance(self, model_name: str, params: Dict[str, Any]):
        """
        Get model instance based on name.
        
        Args:
            model_name: Name of the model
            params: Model hyperparameters
            
        Returns:
            Model instance
        """
        model_map = {
            'LogisticRegression': LogisticRegression,
            'RandomForest': RandomForestClassifier,
            'XGBoost': XGBClassifier,
            'LightGBM': LGBMClassifier
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model_map[model_name](**params)
    
    def train(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            params: Model hyperparameters (optional)
            
        Returns:
            Trained model
        """
        logger.info(f"\nTraining {model_name}...")
        logger.info("-"*60)
        
        self.model_name = model_name
        
        # Get model parameters
        if params is None:
            # Get from config
            if model_name == 'LogisticRegression':
                params = self.config['models']['baseline']['params']
            else:
                params = self.config['models']['tree_based'][model_name]['params']
        
        logger.info(f"Parameters: {params}")
        
        # Initialize model
        self.model = self._get_model_instance(model_name, params)
        
        # Train
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        logger.info(f"✓ Training completed in {self.training_time:.2f}s")
        
        return self.model
    
    def cross_validate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring: str = 'f1'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV scores
        """
        cv_config = self.config.get('cross_validation', {})
        
        if not cv_config.get('enabled', True):
            logger.info("Cross-validation disabled")
            return {}
        
        logger.info("Performing cross-validation...")
        
        n_splits = cv_config.get('n_splits', 10)
        shuffle = cv_config.get('shuffle', True)
        random_state = cv_config.get('random_state', 42)
        
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        
        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        self.cv_scores = cv_scores
        
        cv_results = {
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'cv_min': float(np.min(cv_scores)),
            'cv_max': float(np.max(cv_scores)),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"  CV {scoring}: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
        logger.info(f"  CV range: [{cv_results['cv_min']:.4f}, {cv_results['cv_max']:.4f}]")
        
        return cv_results
    
    def get_model(self):
        """Get trained model."""
        return self.model