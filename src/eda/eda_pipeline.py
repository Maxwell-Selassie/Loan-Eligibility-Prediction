"""
Main EDA Pipeline Orchestrator
Production-grade with comprehensive error handling, logging, and performance optimization.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')
import mlflow

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from eda import (
    DataQuality, DescriptiveStats, Visualizations, InferentialStats
)

from utils import (
    read_csv, write_csv, read_yaml, write_json, save_joblib,
    ensure_directory, get_timestamp, Timer, LoggerMixin
)

class EDAExecutionError(Exception):
    """Custom exception for EDA pipeline errors."""
    pass


class EDAPipeline(LoggerMixin):
    """
    Production-grade EDA Pipeline for Loan Eligibility Prediction.
    
    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        timestamp: Pipeline execution timestamp
        results: Dictionary storing all analysis results
    """
    
    def __init__(self, config_path: str = "config/eda_config.yaml"):
        """
        Initialize EDA Pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.timestamp = get_timestamp()
        self.config = self._load_config(config_path)
        self.logger = self.setup_class_logger('EDA_pipeline', self.config, 'logging')
        self.results: Dict[str, Any] = {}
        self.df: pd.DataFrame | None = None
        
        self.logger.info("="*80)
        self.logger.info(f"EDA PIPELINE INITIALIZED - {self.timestamp}")
        self.logger.info("="*80)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config = read_yaml(config_path)
            return config
        except FileNotFoundError:
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load config: {e}")
            sys.exit(1)
    
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data with optimizations.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            EDAExecutionError: If data loading fails
        """
        with Timer("Data loading", self.logger):
            try:
                data_config = self.config.get('data', {})
                raw_path = data_config.get('raw_path')
                
                self.logger.info(f"Loading data from: {raw_path}")
                
                df = read_csv(
                    filepath=raw_path,
                    optimize_dtypes=data_config.get('optimize_dtypes', True),
                    categorical_columns=data_config.get('categorical_columns', [])
                )
                
                self.df = df
                self.logger.info(f"Data loaded: {len(df):,} rows x {len(df.columns)} columns")

                mlflow.log_artifact('data/raw/LEP.csv')
                
                return df
                
            except Exception as e:
                self.logger.error(f"Data loading failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Failed to load data: {e}")
    

    
    def _generate_summary_report(self, output_file: Path = 'artifacts/summary_report.txt') -> None:
        """Generate human-readable summary report."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"EDA SUMMARY REPORT - {self.timestamp}\n")
                f.write("="*80 + "\n\n")
                
                # Dataset overview
                f.write("DATASET OVERVIEW\n")
                f.write("-"*80 + "\n")
                basic = self.results.get('descriptive_stats', {}).get('basic', {})
                f.write(f"Rows: {basic.get('n_rows', 0):,}\n")
                f.write(f"Columns: {basic.get('n_columns', 0)}\n")
                f.write(f"Memory: {basic.get('memory_usage_mb', 0):.2f} MB\n\n")
                
                # Data quality
                f.write("DATA QUALITY\n")
                f.write("-"*80 + "\n")
                quality = self.results.get('data_quality', {})
                
                missing = quality.get('missing_values', {})
                if missing.get('has_missing'):
                    f.write(f"Missing values: {len(missing.get('missing_df', []))} columns affected\n")
                else:
                    f.write("Missing values: None\n")
                
                duplicates = quality.get('duplicates', {})
                if duplicates.get('has_duplicates'):
                    f.write(f"Duplicates: {duplicates.get('count', 0):,} rows ({duplicates.get('percentage', 0):.2f}%)\n")
                else:
                    f.write("Duplicates: None\n")
                
                f.write("\n")
                
                # Statistical tests
                f.write("STATISTICAL TESTS - SIGNIFICANT RESULTS\n")
                f.write("-"*80 + "\n")
                
                inferential = self.results.get('inferential_stats', {})
                
                # T-tests
                ttests = inferential.get('ttests', {})
                significant_ttests = {k: v for k, v in ttests.items() if v.get('significant')}
                f.write(f"\nT-Tests: {len(significant_ttests)}/{len(ttests)} features significant\n")
                for col, result in significant_ttests.items():
                    f.write(f"  - {col}: p={result['p_value']:.4f}, d={result['cohens_d']:.3f} ({result['effect_interpretation']})\n")
                
                # Chi-square
                chi_tests = inferential.get('chi_square_tests', {})
                significant_chi = {k: v for k, v in chi_tests.items() if v.get('significant')}
                f.write(f"\nChi-Square Tests: {len(significant_chi)}/{len(chi_tests)} features significant\n")
                for col, result in significant_chi.items():
                    f.write(f"  - {col}: p={result['p_value']:.4f}, V={result['cramers_v']:.3f} ({result['effect_interpretation']})\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*80 + "\n")

            mlflow.log_artifact(output_file)
            
            self.logger.info(f"Summary report saved: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")

    
    def execute(self) -> Dict[str, Any]:
        """
        Execute complete EDA pipeline.
        
        Returns:
            Dictionary with all analysis results
            
        Raises:
            EDAExecutionError: If pipeline execution fails
        """

        mlflow.set_tracking_uri(f'sqlite:///loan_eligibility.db')
        mlflow.set_experiment(f'LOAN ELIGIBILITY PREDICTION')


        try:
            with Timer("Complete EDA Pipeline", self.logger):
                with mlflow.start_run(run_name='Complete EDA_Pipeline') as parent_run:
                
                # Load data
                    mlflow.set_tag('component','load_data')
                    df = self.load_data()
                
                    mlflow.set_tag('component','descriptive_stats')
                    DescriptiveStats(self.config).run_descriptive_stats(df)

                    mlflow.set_tag('component','data_quality')
                    DataQuality(self.config).run_data_quality_checks(df)

                    mlflow.set_tag('component', 'inferential_stats')
                    InferentialStats(self.config).compute_confidence_intervals(df)
                    InferentialStats(self.config).run_ttest_parallel(df)
                    InferentialStats(self.config).run_chi_square_tests(df)
                
                    mlflow.set_tag('component','visualizations')
                    Visualizations(self.config).run_visualizations(df)

                    mlflow.set_tag('component','generate_summary_report')
                    self._generate_summary_report()
                
                
                self.logger.info("="*80)
                self.logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY")
                self.logger.info("="*80)
                
                # return self.results
                
        except EDAExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise EDAExecutionError(f"Pipeline failed: {e}")


def main():
    """Main entry point for EDA pipeline."""
    try:
        # Initialize pipeline
        pipeline = EDAPipeline(config_path="config/EDA_config.yaml")
        
        # Execute
        pipeline.execute()
        
        return 0
        
    except EDAExecutionError as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())