"""
Main EDA Pipeline Execution Script
===================================
Production-grade entry point for exploratory data analysis.

Usage:
    python run_eda.py --config config.yaml
    python run_eda.py --config config.yaml --target Loan_Status
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import yaml
import traceback

# Import modules
from utils import (
    setup_logger,
    read_csv,
    write_json,
    get_timestamp,
    create_project_structure,
    optimize_dataframe_memory
)


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_environment(config: dict) -> dict:
    """
    Setup project environment and directories.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with resolved paths
    """
    base_dir = Path(config['paths']['base_dir']).resolve()
    
    # Create directory structure
    dirs = create_project_structure(
        base_dir,
        [
            config['paths']['processed_data'],
            config['paths']['outputs'],
            config['paths']['plots'],
            config['paths']['artifacts'],
            config['paths']['logs'],
            config['paths']['models']
        ]
    )
    
    # Resolve all paths
    paths = {
        'base_dir': base_dir,
        'raw_data': base_dir / config['paths']['raw_data'],
        'processed_data': base_dir / config['paths']['processed_data'],
        'outputs': base_dir / config['paths']['outputs'],
        'plots': base_dir / config['paths']['plots'],
        'artifacts': base_dir / config['paths']['artifacts'],
        'logs': base_dir / config['paths']['logs'],
        'models': base_dir / config['paths']['models']
    }
    
    return paths


def run_eda_pipeline(
    config: dict,
    target_col: Optional[str] = None,
    optimize_memory: bool = True
) -> None:
    """
    Execute complete EDA pipeline.
    
    Args:
        config: Configuration dictionary
        target_col: Override target column from config
        optimize_memory: If True, optimizes DataFrame memory
    """
    # Setup environment
    paths = setup_environment(config)
    
    # Setup logger
    timestamp = get_timestamp()
    log_file = paths['logs'] / f'eda_pipeline_{timestamp}.log'
    logger = setup_logger(
        name='production_eda',
        log_filename=log_file,
        level=config['logging']['level'],
        console_output=config['logging']['console_output'],
        file_rotation=config['logging']['file_rotation']
    )
    
    logger.info("="*70)
    logger.info("STARTING PRODUCTION EDA PIPELINE")
    logger.info("="*70)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Configuration loaded from: config.yaml")
    logger.info(f"Logs: {log_file}")
    
    try:
        # Import EDA class (would normally be from production_eda_pipeline import ProductionEDA)
        from dataclasses import dataclass
        
        @dataclass
        class EDAConfig:
            """Configuration for EDA pipeline."""
            raw_data_path: Path
            output_dir: Path
            plots_dir: Path
            artifacts_dir: Path
            log_file: Path
            confidence_level: float = 0.95
            alpha: float = 0.05
            max_categories_plot: int = 20
            figure_dpi: int = 100
            n_jobs: int = -1
        
        # Create EDA configuration
        eda_config = EDAConfig(
            raw_data_path=paths['raw_data'],
            output_dir=paths['outputs'],
            plots_dir=paths['plots'],
            artifacts_dir=paths['artifacts'],
            log_file=log_file,
            confidence_level=config['eda']['confidence_level'],
            alpha=config['eda']['alpha'],
            max_categories_plot=config['eda']['max_categories_plot'],
            figure_dpi=config['eda']['figure_dpi'],
            n_jobs=config['eda']['n_jobs']
        )
        
        # Load data
        logger.info(f"Loading data from: {paths['raw_data']}")
        df = read_csv(paths['raw_data'])
        logger.info(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Optimize memory if enabled
        if optimize_memory and config['performance']['optimize_memory']:
            logger.info("Optimizing DataFrame memory usage...")
            df = optimize_dataframe_memory(df, verbose=True)
        
        # Get target column
        target = target_col or config['data']['target_column']
        logger.info(f"Target column: {target}")
        
        # Validate target column exists
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        
        # Run EDA pipeline
        logger.info("Initializing EDA pipeline...")
        
        # Since we can't import the full ProductionEDA class in this context,
        # we'll demonstrate the workflow structure
        
        # 1. Basic Statistics
        logger.info("Computing basic statistics...")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if config['data']['id_column'] in numeric_cols:
            numeric_cols.remove(config['data']['id_column'])
        
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        logger.info(f"  Numeric columns: {len(numeric_cols)}")
        logger.info(f"  Categorical columns: {len(categorical_cols)}")
        
        basic_stats = {
            'shape': df.shape,
            'numeric_summary': df[numeric_cols].describe().to_dict(),
            'categorical_summary': {
                col: {
                    'unique': int(df[col].nunique()),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
                for col in categorical_cols
            }
        }
        
        # Save basic statistics
        stats_path = paths['artifacts'] / f'basic_statistics_{timestamp}.json'
        write_json(basic_stats, stats_path)
        logger.info(f"  Basic statistics saved: {stats_path}")
        
        # 2. Missing Values Analysis
        logger.info("Analyzing missing values...")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) > 0:
            logger.info(f"  Found {len(missing)} columns with missing values")
            for col, count in missing.items():
                pct = count / len(df) * 100
                logger.info(f"    {col}: {count} ({pct:.2f}%)")
        else:
            logger.info("  No missing values detected")
        
        # 3. Outlier Detection
        logger.info("Detecting outliers using IQR method...")
        outliers_summary = {}
        
        q1 = df[numeric_cols].quantile(0.25)
        q3 = df[numeric_cols].quantile(0.75)
        iqr = q3 - q1
        
        for col in numeric_cols:
            lower = q1[col] - config['eda']['iqr_multiplier'] * iqr[col]
            upper = q3[col] + config['eda']['iqr_multiplier'] * iqr[col]
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            
            outliers_summary[col] = {
                'n_outliers': int(n_outliers),
                'pct': float(n_outliers / len(df) * 100),
                'lower_bound': float(lower),
                'upper_bound': float(upper)
            }
            
            if n_outliers > 0:
                logger.info(f"  {col}: {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")
        
        # Save outlier analysis
        outliers_path = paths['artifacts'] / f'outliers_{timestamp}.json'
        write_json(outliers_summary, outliers_path)
        logger.info(f"  Outlier analysis saved: {outliers_path}")
        
        # 4. Target Variable Distribution
        logger.info(f"Analyzing target variable: {target}")
        target_dist = df[target].value_counts().to_dict()
        
        for value, count in target_dist.items():
            pct = count / len(df) * 100
            logger.info(f"  {value}: {count} ({pct:.2f}%)")
        
        # Check for class imbalance
        if len(target_dist) == 2:
            values = list(target_dist.values())
            ratio = max(values) / min(values)
            if ratio > 1.5:
                logger.warning(f"  Class imbalance detected! Ratio: {ratio:.2f}:1")
        
        # 5. Summary Report
        logger.info("Generating summary report...")
        
        report_lines = [
            "# Loan Eligibility EDA Report",
            f"\n**Generated**: {timestamp}",
            f"\n## Dataset Overview",
            f"- **Rows**: {df.shape[0]:,}",
            f"- **Columns**: {df.shape[1]}",
            f"- **Numeric Features**: {len(numeric_cols)}",
            f"- **Categorical Features**: {len(categorical_cols)}",
            f"\n## Data Quality",
            f"- **Missing Values**: {len(missing)} columns affected",
            f"- **Outliers**: Detected in {sum(1 for v in outliers_summary.values() if v['n_outliers'] > 0)} columns",
            f"\n## Target Variable: {target}",
        ]
        
        for value, count in target_dist.items():
            pct = count / len(df) * 100
            report_lines.append(f"- **{value}**: {count:,} ({pct:.1f}%)")
        
        report_lines.extend([
            f"\n## Outputs",
            f"- **Artifacts**: `{paths['artifacts']}`",
            f"- **Plots**: `{paths['plots']}`",
            f"- **Logs**: `{log_file}`",
        ])
        
        report_content = "\n".join(report_lines)
        report_path = paths['artifacts'] / f'eda_report_{timestamp}.md'
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"  Report saved: {report_path}")
        
        # Final summary
        logger.info("="*70)
        logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Results saved to: {paths['artifacts']}")
        logger.info(f"Execution log: {log_file}")
        
    except Exception as e:
        logger.error("="*70)
        logger.error("PIPELINE FAILED")
        logger.error("="*70)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


def main():
    """Main entry point for EDA pipeline."""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Production EDA Pipeline for Loan Eligibility Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_eda.py --config config.yaml
  python run_eda.py --config config.yaml --target Loan_Status
  python run_eda.py --config config.yaml --no-optimize
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Target column name (overrides config)'
    )
    
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Disable memory optimization'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='EDA Pipeline v1.0.0'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Run pipeline
        run_eda_pipeline(
            config=config,
            target_col=args.target,
            optimize_memory=not args.no_optimize
        )
        
        print("\n✓ Pipeline completed successfully!")
        print(f"Check outputs directory for results: {config['paths']['outputs']}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        print("Please ensure all required files exist.", file=sys.stderr)
        return 1
        
    except ValueError as e:
        print(f"\n✗ Validation Error: {e}", file=sys.stderr)
        print("Please check your configuration and data.", file=sys.stderr)
        return 1
        
    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}", file=sys.stderr)
        print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())