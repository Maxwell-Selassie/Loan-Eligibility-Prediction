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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    read_csv, write_csv, read_yaml, write_json, save_joblib,
    ensure_directory, get_timestamp, Timer, setup_logger
)
from eda.data_quality import (
    check_missing_values, check_duplicates, detect_outliers_iqr
)
from eda.descriptive_stats import (
    compute_basic_stats, analyze_numeric_columns, analyze_categorical_columns
)
from eda.inferential_stats import (
    compute_confidence_intervals, run_ttest_parallel,
    run_chi_square_tests, multi_group_comparison
)
from eda.visualizations import (
    setup_plot_style, plot_numeric_distributions, plot_boxplots,
    plot_categorical_distributions, plot_correlation_heatmap,
    plot_target_distribution
)


class EDAExecutionError(Exception):
    """Custom exception for EDA pipeline errors."""
    pass


class EDAPipeline:
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
        self.logger = self._setup_logging()
        self.results: Dict[str, Any] = {}
        self.df: Optional[pd.DataFrame] = None
        
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
    
    def _setup_logging(self) -> Any:
        """Setup logging system."""
        log_config = self.config.get('logging', {})
        log_dir = Path(log_config.get('log_dir', '../logs/'))
        
        ensure_directory(log_dir)
        
        logger = setup_logger(
            name='eda_pipeline',
            log_dir=log_dir,
            log_level=log_config.get('log_level', 'INFO'),
            max_bytes=log_config.get('max_bytes', 10485760),
            backup_count=log_config.get('backup_count', 7)
        )
        
        return logger
    
    def _ensure_output_directories(self) -> None:
        """Ensure all output directories exist."""
        self.logger.info("Creating output directories...")
        
        output_config = self.config.get('output', {})
        plots_dir = Path(output_config.get('plots_dir', 'plots/'))
        
        data_config = self.config.get('data', {})
        artifacts_dir = Path(data_config.get('artifacts_path', 'artifacts/'))
        processed_dir = Path(data_config.get('processed_path', 'data/processed/'))
        
        ensure_directory(plots_dir)
        ensure_directory(artifacts_dir)
        ensure_directory(processed_dir)
        
        # Store paths for later use
        self.plots_dir = plots_dir
        self.artifacts_dir = artifacts_dir
        self.processed_dir = processed_dir
    
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
                
                return df
                
            except Exception as e:
                self.logger.error(f"Data loading failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Failed to load data: {e}")
    
    def run_data_quality_checks(self) -> Dict[str, Any]:
        """
        Execute data quality checks.
        
        Returns:
            Dictionary with quality check results
        """
        self.logger.info("="*80)
        self.logger.info("DATA QUALITY CHECKS")
        self.logger.info("="*80)
        
        quality_results = {}
        
        with Timer("Data quality checks", self.logger):
            try:
                dq_config = self.config.get('data_quality', {})
                
                # Missing values
                missing_result = check_missing_values(
                    self.df,
                    warning_threshold=dq_config.get('missing_threshold_warning', 0.05),
                    critical_threshold=dq_config.get('missing_threshold_critical', 0.30)
                )
                quality_results['missing_values'] = missing_result
                
                # Duplicates
                if dq_config.get('check_duplicates', True):
                    duplicate_result = check_duplicates(self.df)
                    quality_results['duplicates'] = duplicate_result
                
                # Outliers
                if dq_config.get('detect_outliers', True):
                    numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                    id_col = self.config['data'].get('id_column')
                    
                    outlier_result = detect_outliers_iqr(
                        self.df,
                        numeric_cols,
                        multiplier=self.config['statistics'].get('outlier_iqr_multiplier', 1.5),
                        id_column=id_col
                    )
                    quality_results['outliers'] = outlier_result
                
                self.results['data_quality'] = quality_results
                return quality_results
                
            except Exception as e:
                self.logger.error(f"Data quality checks failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Data quality checks failed: {e}")
    
    def run_descriptive_analysis(self) -> Dict[str, Any]:
        """
        Execute descriptive statistical analysis.
        
        Returns:
            Dictionary with descriptive statistics
        """
        self.logger.info("="*80)
        self.logger.info("DESCRIPTIVE STATISTICS")
        self.logger.info("="*80)
        
        descriptive_results = {}
        
        with Timer("Descriptive analysis", self.logger):
            try:
                # Basic stats
                basic_stats = compute_basic_stats(self.df)
                descriptive_results['basic'] = basic_stats
                
                # Numeric columns
                id_col = self.config['data'].get('id_column')
                numeric_summary, numeric_cols = analyze_numeric_columns(self.df, id_col)
                descriptive_results['numeric_summary'] = numeric_summary.to_dict()
                descriptive_results['numeric_columns'] = numeric_cols
                
                # Categorical columns
                target_col = self.config['data'].get('target_column')
                cat_summary, cat_cols = analyze_categorical_columns(self.df, target_col)
                descriptive_results['categorical_summary'] = cat_summary.to_dict()
                descriptive_results['categorical_columns'] = cat_cols
                
                # Target variable distribution
                if target_col and target_col in self.df.columns:
                    target_dist = self.df[target_col].value_counts().to_dict()
                    descriptive_results['target_distribution'] = target_dist
                    
                    self.logger.info(f"Target variable '{target_col}' distribution:")
                    for value, count in target_dist.items():
                        pct = (count / len(self.df)) * 100
                        self.logger.info(f"  {value}: {count:,} ({pct:.2f}%)")
                
                self.results['descriptive_stats'] = descriptive_results
                return descriptive_results
                
            except Exception as e:
                self.logger.error(f"Descriptive analysis failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Descriptive analysis failed: {e}")
    
    def run_inferential_analysis(self) -> Dict[str, Any]:
        """
        Execute inferential statistical analysis.
        
        Returns:
            Dictionary with inferential statistics
        """
        self.logger.info("="*80)
        self.logger.info("INFERENTIAL STATISTICS")
        self.logger.info("="*80)
        
        inferential_results = {}
        
        with Timer("Inferential analysis", self.logger):
            try:
                stats_config = self.config.get('statistics', {})
                data_config = self.config.get('data', {})
                
                target_col = data_config.get('target_column')
                id_col = data_config.get('id_column')
                positive_val = data_config.get('target_positive_value', 'Y')
                negative_val = data_config.get('target_negative_value', 'N')
                
                # Get column lists
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                if id_col in numeric_cols:
                    numeric_cols.remove(id_col)
                
                cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
                if target_col in cat_cols:
                    cat_cols.remove(target_col)
                
                # Confidence intervals
                self.logger.info("-" * 60)
                self.logger.info("Computing Confidence Intervals")
                self.logger.info("-" * 60)
                
                ci_results = compute_confidence_intervals(
                    self.df,
                    numeric_cols,
                    confidence=stats_config.get('confidence_level', 0.95)
                )
                inferential_results['confidence_intervals'] = ci_results
                
                # T-tests (parallel execution)
                if target_col and target_col in self.df.columns:
                    self.logger.info("-" * 60)
                    self.logger.info("T-Tests: Numeric Features vs Target")
                    self.logger.info("-" * 60)
                    
                    perf_config = self.config.get('performance', {})
                    n_jobs = perf_config.get('n_jobs', -1) if perf_config.get('parallel_statistical_tests', True) else 1
                    
                    ttest_results = run_ttest_parallel(
                        self.df,
                        numeric_cols,
                        target_col,
                        positive_val,
                        negative_val,
                        n_jobs=n_jobs
                    )
                    inferential_results['ttests'] = ttest_results
                    
                    # Log significant results
                    significant_features = [
                        col for col, result in ttest_results.items() 
                        if result['significant']
                    ]
                    self.logger.info(f"Significant features ({len(significant_features)}): {significant_features}")
                    
                    # Chi-square tests
                    if cat_cols:
                        self.logger.info("-" * 60)
                        self.logger.info("Chi-Square Tests: Categorical Features vs Target")
                        self.logger.info("-" * 60)
                        
                        chi_square_results = run_chi_square_tests(
                            self.df,
                            cat_cols,
                            target_col,
                            alpha=stats_config.get('significance_level', 0.05)
                        )
                        inferential_results['chi_square_tests'] = chi_square_results
                
                self.results['inferential_stats'] = inferential_results
                return inferential_results
                
            except Exception as e:
                self.logger.error(f"Inferential analysis failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Inferential analysis failed: {e}")
    
    def generate_visualizations(self) -> None:
        """Generate all visualizations."""
        self.logger.info("="*80)
        self.logger.info("GENERATING VISUALIZATIONS")
        self.logger.info("="*80)
        
        with Timer("Visualization generation", self.logger):
            try:
                viz_config = self.config.get('visualization', {})
                output_config = self.config.get('output', {})
                
                # Setup plot style
                setup_plot_style(
                    style=viz_config.get('style', 'seaborn-v0_8-darkgrid'),
                    context=viz_config.get('context', 'notebook')
                )
                
                dpi = output_config.get('plot_dpi', 300)
                
                # Get column lists
                id_col = self.config['data'].get('id_column')
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                if id_col in numeric_cols:
                    numeric_cols.remove(id_col)
                
                cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
                target_col = self.config['data'].get('target_column')
                
                # Numeric distributions
                if numeric_cols:
                    plot_numeric_distributions(
                        self.df,
                        numeric_cols,
                        self.plots_dir,
                        figsize=tuple(viz_config.get('figure_size_univariate', [18, 10])),
                        dpi=dpi,
                        kde=viz_config.get('histogram', {}).get('kde', True)
                    )
                
                # Boxplots
                if numeric_cols:
                    plot_boxplots(
                        self.df,
                        numeric_cols,
                        self.plots_dir,
                        figsize=tuple(viz_config.get('figure_size_univariate', [18, 10])),
                        dpi=dpi
                    )
                
                # Categorical distributions
                if cat_cols:
                    plot_categorical_distributions(
                        self.df,
                        cat_cols,
                        self.plots_dir,
                        figsize=tuple(viz_config.get('figure_size_multivariate', [15, 10])),
                        dpi=dpi
                    )
                
                # Correlation heatmap
                if len(numeric_cols) >= 2:
                    plot_correlation_heatmap(
                        self.df,
                        numeric_cols,
                        self.plots_dir,
                        method='spearman',
                        figsize=tuple(viz_config.get('figure_size_multivariate', [15, 10])),
                        dpi=dpi
                    )
                
                # Target distribution
                if target_col and target_col in self.df.columns:
                    plot_target_distribution(
                        self.df,
                        target_col,
                        self.plots_dir,
                        figsize=(8, 6),
                        dpi=dpi
                    )
                
                self.logger.info(f"All visualizations saved to: {self.plots_dir}")
                
            except Exception as e:
                self.logger.error(f"Visualization generation failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Visualization generation failed: {e}")
    
    def save_artifacts(self) -> None:
        """Save all analysis artifacts."""
        self.logger.info("="*80)
        self.logger.info("SAVING ARTIFACTS")
        self.logger.info("="*80)
        
        with Timer("Artifact saving", self.logger):
            try:
                output_config = self.config.get('output', {})
                
                if not output_config.get('save_artifacts', True):
                    self.logger.info("Artifact saving disabled in config")
                    return
                
                artifact_format = output_config.get('artifact_format', 'joblib')
                
                # Add metadata
                self.results['metadata'] = {
                    'timestamp': self.timestamp,
                    'n_rows': len(self.df),
                    'n_columns': len(self.df.columns),
                    'columns': self.df.columns.tolist()
                }
                
                # Save based on format
                if artifact_format == 'joblib':
                    output_file = self.artifacts_dir / f"eda_results_{self.timestamp}.joblib"
                    save_joblib(self.results, output_file, compress=3)
                    self.logger.info(f"Saved artifacts (joblib): {output_file}")
                
                elif artifact_format == 'json':
                    output_file = self.artifacts_dir / f"eda_results_{self.timestamp}.json"
                    write_json(self.results, output_file, indent=2)
                    self.logger.info(f"Saved artifacts (JSON): {output_file}")
                
                # Also save summary report
                summary_file = self.artifacts_dir / f"eda_summary_{self.timestamp}.txt"
                self._generate_summary_report(summary_file)
                
            except Exception as e:
                self.logger.error(f"Artifact saving failed: {e}", exc_info=True)
                raise EDAExecutionError(f"Artifact saving failed: {e}")
    
    def _generate_summary_report(self, output_file: Path) -> None:
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
        try:
            with Timer("Complete EDA Pipeline", self.logger):
                # Setup
                self._ensure_output_directories()
                
                # Load data
                self.load_data()
                
                # Analysis stages
                self.run_data_quality_checks()
                self.run_descriptive_analysis()
                self.run_inferential_analysis()
                
                # Visualizations
                self.generate_visualizations()
                
                # Save results
                self.save_artifacts()
                
                self.logger.info("="*80)
                self.logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY")
                self.logger.info("="*80)
                
                return self.results
                
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
        results = pipeline.execute()
        
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