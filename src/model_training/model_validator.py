"""
Model validation module for production safety checks.
"""

import logging
from typing import Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validate models before deployment with safety checks.
    
    Validates:
    - Performance thresholds
    - Overfitting (train-val gap)
    - Business rules
    - Comparison with production model
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelValidator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('model_validation', {})
        self.registry_config = config.get('mlflow', {}).get('model_registry', {})
        self.client = MlflowClient()
    
    def validate_performance(
        self,
        metrics: Dict[str, float],
        stage: str = 'val'
    ) -> Dict[str, Any]:
        """
        Validate model performance against thresholds.
        
        Args:
            metrics: Model metrics (f1_score, roc_auc, etc.)
            stage: 'train' or 'val'
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating {stage} performance...")
        
        thresholds = self.config.get('thresholds', {})
        results = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'checks': {}
        }
        
        for metric_name, metric_value in metrics.items():
            if metric_name not in thresholds:
                continue
            
            threshold_config = thresholds[metric_name]
            min_threshold = threshold_config.get('min', 0)
            warning_threshold = threshold_config.get('warning', 0)
            target_threshold = threshold_config.get('target', 1)
            
            check_result = {
                'value': metric_value,
                'min': min_threshold,
                'warning': warning_threshold,
                'target': target_threshold,
                'status': 'PASS'
            }
            
            # Check minimum
            if metric_value < min_threshold:
                check_result['status'] = 'FAIL'
                results['passed'] = False
                results['errors'].append(
                    f"{metric_name}={metric_value:.4f} below minimum {min_threshold}"
                )
                logger.error(f"✗ {metric_name}: {metric_value:.4f} < {min_threshold} (FAIL)")
            
            # Check warning
            elif metric_value < warning_threshold:
                check_result['status'] = 'WARNING'
                results['warnings'].append(
                    f"{metric_name}={metric_value:.4f} below warning threshold {warning_threshold}"
                )
                logger.warning(f"⚠ {metric_name}: {metric_value:.4f} < {warning_threshold} (WARNING)")
            
            # Check target
            elif metric_value >= target_threshold:
                check_result['status'] = 'EXCELLENT'
                logger.info(f"✓ {metric_name}: {metric_value:.4f} >= {target_threshold} (EXCELLENT)")
            
            else:
                logger.info(f"✓ {metric_name}: {metric_value:.4f} (PASS)")
            
            results['checks'][metric_name] = check_result
        
        return results
    
    def validate_overfitting(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        primary_metric: str = 'f1_score'
    ) -> Dict[str, Any]:
        """
        Check for overfitting (train-val gap).
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            primary_metric: Primary metric to check
            
        Returns:
            Validation results
        """
        logger.info("Checking for overfitting...")
        
        train_score = train_metrics.get(primary_metric, 0)
        val_score = val_metrics.get(primary_metric, 0)
        
        gap = train_score - val_score
        gap_pct = (gap / train_score) * 100 if train_score > 0 else 0
        
        max_gap = self.config.get('thresholds', {}).get('train_test_gap', {}).get('max', 0.15)
        warning_gap = self.config.get('thresholds', {}).get('train_test_gap', {}).get('warning', 0.10)
        
        result = {
            'gap': gap,
            'gap_percentage': gap_pct,
            'max_allowed': max_gap,
            'passed': gap <= max_gap,
            'status': 'PASS'
        }
        
        if gap > max_gap:
            result['status'] = 'FAIL'
            result['passed'] = False
            logger.error(f"✗ Overfitting detected: gap={gap:.4f} ({gap_pct:.1f}%) > max={max_gap}")
        elif gap > warning_gap:
            result['status'] = 'WARNING'
            logger.warning(f"⚠ Moderate overfitting: gap={gap:.4f} ({gap_pct:.1f}%)")
        else:
            logger.info(f"✓ No overfitting: gap={gap:.4f} ({gap_pct:.1f}%)")
        
        return result
    
    def validate_business_rules(
        self,
        metrics: Dict[str, float],
        confusion_matrix: Any
    ) -> Dict[str, Any]:
        """
        Validate business-specific rules.
        
        Args:
            metrics: Model metrics
            confusion_matrix: Confusion matrix
            
        Returns:
            Validation results
        """
        logger.info("Validating business rules...")
        
        business_rules = self.config.get('business_rules', {})
        results = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'checks': {}
        }
        
        # Check recall for positive class (don't reject good applicants)
        min_recall = business_rules.get('min_recall_for_positive_class', 0.80)
        actual_recall = metrics.get('recall', 0)
        
        recall_check = {
            'value': actual_recall,
            'min': min_recall,
            'passed': actual_recall >= min_recall
        }
        
        if actual_recall < min_recall:
            recall_check['status'] = 'FAIL'
            results['passed'] = False
            results['errors'].append(
                f"Recall={actual_recall:.4f} below minimum {min_recall}"
            )
            logger.error(f"✗ Recall: {actual_recall:.4f} < {min_recall} (rejecting too many good applicants)")
        else:
            recall_check['status'] = 'PASS'
            logger.info(f"✓ Recall: {actual_recall:.4f} >= {min_recall}")
        
        results['checks']['recall'] = recall_check
        
        # Check false positive rate (don't approve bad loans)
        if confusion_matrix is not None:
            cm = np.array(confusion_matrix)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            max_fpr = business_rules.get('max_false_positive_rate', 0.30)
            
            fpr_check = {
                'value': fpr,
                'max': max_fpr,
                'passed': fpr <= max_fpr
            }
            
            if fpr > max_fpr:
                fpr_check['status'] = 'FAIL'
                results['passed'] = False
                results['errors'].append(
                    f"False Positive Rate={fpr:.4f} above maximum {max_fpr}"
                )
                logger.error(f"✗ FPR: {fpr:.4f} > {max_fpr} (approving too many bad loans)")
            else:
                fpr_check['status'] = 'PASS'
                logger.info(f"✓ FPR: {fpr:.4f} <= {max_fpr}")
            
            results['checks']['false_positive_rate'] = fpr_check
        
        return results
    
    def compare_with_production(
        self,
        new_model_metrics: Dict[str, float],
        primary_metric: str = 'f1_score'
    ) -> Dict[str, Any]:
        """
        Compare new model with current production model.
        
        Args:
            new_model_metrics: New model validation metrics
            primary_metric: Primary metric for comparison
            
        Returns:
            Comparison results
        """
        if not self.registry_config.get('compare_with_production', True):
            logger.info("Production comparison disabled")
            return {'should_replace': True, 'reason': 'comparison_disabled'}
        
        logger.info("Comparing with production model...")
        
        try:
            model_name = self.registry_config.get('model_name', 'LoanEligibilityClassifier')
            
            # Get production model
            prod_versions = self.client.get_latest_versions(model_name, stages=["Production"])
            
            if not prod_versions:
                logger.info("No production model found. New model will be first production model.")
                return {
                    'should_replace': True,
                    'reason': 'no_production_model',
                    'new_model_score': new_model_metrics.get(primary_metric, 0)
                }
            
            prod_version = prod_versions[0]
            prod_run_id = prod_version.run_id
            
            # Get production model metrics
            prod_run = self.client.get_run(prod_run_id)
            prod_score = prod_run.data.metrics.get(f"val_{primary_metric}", 0)
            
            new_score = new_model_metrics.get(primary_metric, 0)
            improvement = new_score - prod_score
            improvement_pct = (improvement / prod_score) * 100 if prod_score > 0 else 0
            
            min_improvement = self.registry_config.get('min_improvement_required', 0.01)
            
            result = {
                'production_score': prod_score,
                'new_model_score': new_score,
                'improvement': improvement,
                'improvement_percentage': improvement_pct,
                'min_improvement_required': min_improvement,
                'should_replace': improvement >= min_improvement
            }
            
            if improvement >= min_improvement:
                logger.info(f"✓ New model better: {new_score:.4f} vs {prod_score:.4f} (+{improvement_pct:.1f}%)")
                result['reason'] = 'better_performance'
            else:
                logger.warning(f"✗ New model not better enough: {new_score:.4f} vs {prod_score:.4f} ({improvement_pct:+.1f}%)")
                result['reason'] = 'insufficient_improvement'
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compare with production: {e}")
            return {
                'should_replace': False,
                'reason': 'comparison_failed',
                'error': str(e)
            }
    
    def validate_model(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        confusion_matrix: Any,
        primary_metric: str = 'f1_score'
    ) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            confusion_matrix: Confusion matrix
            primary_metric: Primary metric
            
        Returns:
            Complete validation results
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL VALIDATION")
        logger.info("="*80)
        
        validation_results = {
            'overall_passed': True,
            'checks': {}
        }
        
        # 1. Validate validation performance
        perf_validation = self.validate_performance(val_metrics, stage='val')
        validation_results['checks']['performance'] = perf_validation
        
        if not perf_validation['passed']:
            validation_results['overall_passed'] = False
        
        # 2. Check overfitting
        overfitting_check = self.validate_overfitting(train_metrics, val_metrics, primary_metric)
        validation_results['checks']['overfitting'] = overfitting_check
        
        if not overfitting_check['passed']:
            validation_results['overall_passed'] = False
        
        # 3. Validate business rules
        business_validation = self.validate_business_rules(val_metrics, confusion_matrix)
        validation_results['checks']['business_rules'] = business_validation
        
        if not business_validation['passed']:
            validation_results['overall_passed'] = False
        
        # 4. Compare with production
        prod_comparison = self.compare_with_production(val_metrics, primary_metric)
        validation_results['checks']['production_comparison'] = prod_comparison
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        
        if validation_results['overall_passed']:
            logger.info("✓ ALL VALIDATION CHECKS PASSED")
        else:
            logger.error("✗ VALIDATION FAILED")
            
            # Log all errors
            for check_name, check_result in validation_results['checks'].items():
                if isinstance(check_result, dict) and not check_result.get('passed', True):
                    logger.error(f"  Failed check: {check_name}")
                    if 'errors' in check_result:
                        for error in check_result['errors']:
                            logger.error(f"    - {error}")
        
        return validation_results