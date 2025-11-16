"""
Compare staging model with production model before promotion.
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import Dict, Any
import logging


logger = logging.getLogger(__name__)


def compare_models(
    model_name: str,
    tracking_uri: str = './mlruns'
) -> Dict[str, Any]:
    """
    Compare Staging model with Production model.
    
    Args:
        model_name: Name of registered model
        tracking_uri: MLflow tracking URI
        
    Returns:
        Comparison results
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    logger.info("="*80)
    logger.info("MODEL COMPARISON: Staging vs Production")
    logger.info("="*80)
    
    # Get staging model
    staging_models = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_models:
        logger.error("No model in Staging stage")
        return {}
    
    staging_model = staging_models[0]
    staging_run = client.get_run(staging_model.run_id)
    
    # Get production model
    prod_models = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_models:
        logger.warning("No model in Production stage")
        prod_model = None
        prod_run = None
    else:
        prod_model = prod_models[0]
        prod_run = client.get_run(prod_model.run_id)
    
    # Metrics to compare
    metrics_to_compare = [
        'val_f1_score',
        'val_precision',
        'val_recall',
        'val_roc_auc',
        'train_val_gap',
        'training_time'
    ]
    
    comparison = {
        'staging': {},
        'production': {},
        'differences': {}
    }
    
    logger.info("\nStaging Model:")
    logger.info(f"  Version: {staging_model.version}")
    logger.info(f"  Run ID: {staging_model.run_id}")
    
    for metric in metrics_to_compare:
        value = staging_run.data.metrics.get(metric, 0)
        comparison['staging'][metric] = value
        logger.info(f"  {metric}: {value:.4f}")
    
    if prod_run:
        logger.info("\nProduction Model:")
        logger.info(f"  Version: {prod_model.version}")
        logger.info(f"  Run ID: {prod_model.run_id}")
        
        for metric in metrics_to_compare:
            value = prod_run.data.metrics.get(metric, 0)
            comparison['production'][metric] = value
            logger.info(f"  {metric}: {value:.4f}")
        
        # Calculate differences
        logger.info("\nDifferences (Staging - Production):")
        for metric in metrics_to_compare:
            staging_val = comparison['staging'][metric]
            prod_val = comparison['production'][metric]
            diff = staging_val - prod_val
            diff_pct = (diff / prod_val * 100) if prod_val != 0 else 0
            
            comparison['differences'][metric] = {
                'absolute': diff,
                'percentage': diff_pct
            }
            
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            logger.info(f"  {metric}: {diff:+.4f} ({diff_pct:+.1f}%) {symbol}")
        
        # Recommendation
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDATION")
        logger.info("="*80)
        
        staging_f1 = comparison['staging']['val_f1_score']
        prod_f1 = comparison['production']['val_f1_score']
        improvement = staging_f1 - prod_f1
        
        if improvement > 0.01:  # 1% improvement
            logger.info("✓ RECOMMEND PROMOTION")
            logger.info(f"  Staging model shows {improvement:.4f} ({improvement/prod_f1*100:.1f}%) improvement")
        elif improvement > 0:
            logger.info("⚠ MARGINAL IMPROVEMENT")
            logger.info(f"  Staging model slightly better, but improvement is small")
            logger.info("  Consider A/B testing before full promotion")
        else:
            logger.info("✗ DO NOT PROMOTE")
            logger.info(f"  Staging model is worse than production")
    
    return comparison


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Staging vs Production models')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--tracking-uri', default='./mlruns', help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    compare_models(
        model_name=args.model_name,
        tracking_uri=args.tracking_uri
    )


if __name__ == "__main__":
    main()