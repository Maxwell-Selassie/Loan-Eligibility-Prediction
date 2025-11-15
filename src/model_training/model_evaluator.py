"""
Comprehensive model evaluation with multiple metrics.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, brier_score_loss, log_loss
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate trained models with comprehensive metrics.
    
    Attributes:
        config: Configuration dictionary
        metrics_results: Dictionary storing evaluation results
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelEvaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('metrics', {})
        self.metrics_results = {}
    
    def evaluate(
        self,
        model,
        X: np.ndarray,
        y_true: np.ndarray,
        dataset_name: str = 'validation'
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X: Features
            y_true: True labels
            dataset_name: Name of dataset (train/val/test)
            
        Returns:
            Dictionary with all metrics
        """
        logger.info(f"\nEvaluating on {dataset_name} set...")
        logger.info("-"*60)
        
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        results = {}
        
        # Classification metrics
        classification_metrics = self.config.get('classification_metrics', [])
        
        for metric_name in classification_metrics:
            if metric_name == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric_name == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric_name == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric_name == 'f1_score':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric_name == 'roc_auc':
                score = roc_auc_score(y_true, y_pred_proba)
            elif metric_name == 'average_precision':
                score = average_precision_score(y_true, y_pred_proba)
            else:
                continue
            
            results[metric_name] = float(score)
            logger.info(f"  {metric_name}: {score:.4f}")
        
        # Calibration metrics
        if self.config.get('calibration_metrics', {}).get('enabled', True):
            brier = brier_score_loss(y_true, y_pred_proba)
            logloss = log_loss(y_true, y_pred_proba)
            
            results['brier_score'] = float(brier)
            results['log_loss'] = float(logloss)
            
            logger.info(f"  brier_score: {brier:.4f}")
            logger.info(f"  log_loss: {logloss:.4f}")
        
        # Confusion matrix
        if self.config.get('confusion_matrix', {}).get('enabled', True):
            cm = confusion_matrix(y_true, y_pred)
            normalize = self.config['confusion_matrix'].get('normalize', None)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            results['confusion_matrix'] = cm.tolist()
            
            logger.info(f"\n  Confusion Matrix:")
            logger.info(f"    {cm}")
        
        # Classification report
        if self.config.get('classification_report', {}).get('enabled', True):
            report = classification_report(
                y_true, y_pred,
                output_dict=True,
                zero_division=0
            )
            results['classification_report'] = report
            
            logger.info(f"\n  Per-class metrics:")
            for class_label in ['0', '1']:
                if class_label in report:
                    logger.info(f"    Class {class_label}:")
                    logger.info(f"      Precision: {report[class_label]['precision']:.4f}")
                    logger.info(f"      Recall: {report[class_label]['recall']:.4f}")
                    logger.info(f"      F1-score: {report[class_label]['f1-score']:.4f}")
        
        self.metrics_results[dataset_name] = results
        
        return results
    
    def compare_train_val(
        self,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare training and validation metrics for generalization.
        
        Args:
            train_metrics: Training set metrics
            val_metrics: Validation set metrics
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("\nGeneralization Analysis:")
        logger.info("-"*60)
        
        primary_metric = self.config.get('primary_metric', 'f1_score')
        
        train_score = train_metrics.get(primary_metric, 0)
        val_score = val_metrics.get(primary_metric, 0)
        
        gap = train_score - val_score
        gap_pct = (gap / train_score * 100) if train_score > 0 else 0
        
        logger.info(f"  Train {primary_metric}: {train_score:.4f}")
        logger.info(f"  Val {primary_metric}: {val_score:.4f}")
        logger.info(f"  Gap: {gap:.4f} ({gap_pct:.2f}%)")
        
        if gap > 0.10:
            logger.warning(f"  ⚠ Large gap detected - possible overfitting!")
        elif gap < -0.05:
            logger.warning(f"  ⚠ Negative gap - unusual, check data!")
        else:
            logger.info(f"  ✓ Acceptable generalization")
        
        return {
            'train_score': train_score,
            'val_score': val_score,
            'gap': gap,
            'gap_percentage': gap_pct,
            'generalization_status': 'good' if abs(gap) <= 0.10 else 'poor'
        }