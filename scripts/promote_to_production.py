"""
Manual script to promote model from Staging to Production.
Use this after validating model in staging environment.
"""

import mlflow
from mlflow.tracking import MlflowClient
import argparse
import logging

logger = logging.getLogger(__name__)


def promote_to_production(
    model_name: str,
    version: int,
    tracking_uri: str = './mlruns'
):
    """
    Promote model from Staging to Production.
    
    Args:
        model_name: Name of registered model
        version: Model version to promote
        tracking_uri: MLflow tracking URI
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    logger.info("="*80)
    logger.info("MODEL PROMOTION TO PRODUCTION")
    logger.info("="*80)
    
    try:
        # Get model info
        model_version = client.get_model_version(model_name, version)
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Version: {version}")
        logger.info(f"Current stage: {model_version.current_stage}")
        logger.info(f"Run ID: {model_version.run_id}")
        
        # Get metrics
        run = client.get_run(model_version.run_id)
        val_f1 = run.data.metrics.get('val_f1_score', 0)
        val_auc = run.data.metrics.get('val_roc_auc', 0)
        
        logger.info(f"Val F1-Score: {val_f1:.4f}")
        logger.info(f"Val ROC-AUC: {val_auc:.4f}")
        
        # Confirmation prompt
        print("\n" + "="*80)
        print("⚠️  WARNING: You are about to promote this model to PRODUCTION")
        print("="*80)
        print("This will:")
        print("1. Archive current production model")
        print("2. Make this model the new production model")
        print("3. Affect live predictions")
        print("\nHave you:")
        print("✓ Tested the model in staging?")
        print("✓ Validated performance on real data?")
        print("✓ Run A/B tests (if applicable)?")
        print("✓ Obtained approval from stakeholders?")
        print("="*80)
        
        confirm = input("\nType 'PROMOTE' to confirm: ")
        
        if confirm != 'PROMOTE':
            logger.info("Promotion cancelled by user")
            return
        
        # Archive current production models
        prod_models = client.get_latest_versions(model_name, stages=["Production"])
        for prod_model in prod_models:
            logger.info(f"Archiving production version {prod_model.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=prod_model.version,
                stage="Archived",
                archive_existing_versions=False
            )
        
        # Promote new model to production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        logger.info("="*80)
        logger.info("✓ MODEL SUCCESSFULLY PROMOTED TO PRODUCTION")
        logger.info("="*80)
        logger.info(f"Model: {model_name} v{version}")
        logger.info(f"New stage: Production")
        logger.info("Monitor closely for next 24-48 hours")
        
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Promote MLflow model to Production')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--version', type=int, required=True, help='Model version')
    parser.add_argument('--tracking-uri', default='./mlruns', help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    promote_to_production(
        model_name=args.model_name,
        version=args.version,
        tracking_uri=args.tracking_uri
    )


if __name__ == "__main__":
    main()