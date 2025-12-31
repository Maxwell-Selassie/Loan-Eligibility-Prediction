"""
Data quality checks: missing values, duplicates, outliers.
Optimized for performance and comprehensive reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
from utils import LoggerMixin, write_json, ensure_directory
import mlflow

class DataQuality(LoggerMixin):

    def __init__(self, config):
        self.config = config
        self.logger = self.setup_class_logger('Data_Quality', config, 'logging')

    def check_missing_values(self,
        df: pd.DataFrame,

    ) -> Dict[str, any]:
        """
        Analyze missing values with thresholds.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with missing value analysis
        """

        warning_threshold: float = self.config['data_quality'].get('missing_threshold_warning', 0.05),
        critical_threshold: float = self.config['data_quality'].get('missing_threshold_critical', 0.30)

        self.logger.info("Analyzing missing values...")
        
        missing = df.isnull().sum()
        mlflow.log_metric('total_missing',missing.sum())
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            self.logger.info("No missing values found")
            return {"has_missing": False, "missing_df": pd.DataFrame()}
        
        total_rows = len(df)
        missing_pct = (missing / total_rows).round(4)
        
        missing_df = pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct,
            'severity': ['CRITICAL' if pct >= critical_threshold 
                        else 'WARNING' if pct >= warning_threshold 
                        else 'INFO' 
                        for pct in missing_pct]
        })
        
        # Log summary
        critical_cols = missing_df[missing_df['severity'] == 'CRITICAL']
        warning_cols = missing_df[missing_df['severity'] == 'WARNING']

        mlflow.log_metric('total_critical_cols', len(critical_cols))
        mlflow.log_metric('total_warning_cols', len(warning_cols))
        
        self.logger.info(f"Missing values found in {len(missing_df)} columns")
        if len(critical_cols) > 0:
            self.logger.warning(f"CRITICAL: {len(critical_cols)} columns exceed {critical_threshold*100}% missing")
        if len(warning_cols) > 0:
            self.logger.warning(f"WARNING: {len(warning_cols)} columns exceed {warning_threshold*100}% missing")
        
        return {
            "has_missing": True,
            "missing_df": missing_df,
            "critical_columns": critical_cols.index.tolist(),
            "warning_columns": warning_cols.index.tolist()
        }


    def check_duplicates(self,df: pd.DataFrame) -> Dict[str, any]:
        """
        Check for duplicate rows.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with duplicate analysis
        """
        if self.config['data_quality']['check_duplicates']:
            self.logger.info("Checking for duplicate rows...")
            
            duplicates = df.duplicated(keep='first')
            n_duplicates = duplicates.sum()
            mlflow.log_metric('n_duplicates', n_duplicates)
            
            if n_duplicates == 0:
                self.logger.info("No duplicate rows found")
                return {"has_duplicates": False, "count": 0, "percentage": 0.0}
            
            duplicate_pct = (n_duplicates / len(df)) * 100
            self.logger.warning(f"Found {n_duplicates:,} duplicate rows ({duplicate_pct:.2f}%)")
            
            return {
                "has_duplicates": True,
                "count": n_duplicates,
                "percentage": duplicate_pct,
                "duplicate_rows": df[duplicates]
            }
        else:
            self.logger.warning(f'Duplicates checking disabled! (Skipping...)')
            return {}


    def detect_outliers_iqr(self,
        df: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Detect outliers using IQR method (vectorized for speed).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with outlier analysis per column
        """

        multiplier: float = self.config['statistics'].get('outlier_iqr_multiplier', 1.5),
        id_column: str | None = self.config['data'].get('id_column', None)

        if self.config['data_quality']['detect_outliers'] and self.config['data_quality']['outlier_report']:
            self.logger.info(f"Detecting outliers using IQR method (multiplier={multiplier})...")

            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Filter out ID column
            cols_to_check = [col for col in numeric_columns if col != id_column]
            outlier_summary = {}
            
            # Vectorized computation
            for col in cols_to_check:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                # Vectorized outlier detection
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                n_outliers = outlier_mask.sum()
                outlier_pct = (n_outliers / len(df)) * 100
                mlflow.log_metric('n_outliers', n_outliers)
                mlflow.log_metric('outlier_pct', outlier_pct)
                
                outlier_summary[col] = {
                    'count': int(n_outliers),
                    'percentage': round(outlier_pct, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                    'Q1': round(Q1, 2),
                    'Q3': round(Q3, 2),
                    'IQR': round(IQR, 2)
                }
                
                self.logger.info(f"{col}: {n_outliers} outliers ({outlier_pct:.2f}%) | Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                self.logger.info(f'Total Number of outliers in the dataset : {outlier_mask.sum().sum()}')
            
            return outlier_summary
        else:
            self.logger.warning(f'Outlier detection disabled! (Skipping...)')
            return {}
        

    def run_data_quality_checks(self, df):
        '''Runs all data quality checks and logs necessary info to MLflow'''
        # missing values
        missing_results =  self.check_missing_values(df)
        MISSING_PATH =  Path('artifacts/data/missing_values.json')
        try:
            write_json(missing_results, MISSING_PATH,indent=4)
            self.logger.info(f'Data successfully saved to {MISSING_PATH}')
        except Exception as e:
            self.logger.error(f'An error occured during JSON operation: {e}')
            raise
        mlflow.log_artifact(MISSING_PATH)

        # duplicate values
        duplicates_result = self.check_duplicates(df)
        DUPLICATES_PATH = Path('artifacts/data/duplicates_data.json')
        try:
            write_json(duplicates_result, DUPLICATES_PATH,indent=4)
            self.logger.info(f'Data successfully saved to {DUPLICATES_PATH}')
        except Exception as e:
            self.logger.error(f'An error occured during JSON operation: {e}')
            raise
        mlflow.log_artifact(DUPLICATES_PATH)

        # outlier values
        outliers_result = self.detect_outliers_iqr(df)
        OUTLIERS_PATH = Path('artifacts/data/outliers_data.json')
        try:
            write_json(outliers_result, OUTLIERS_PATH,indent=4)
            self.logger.info(f'Data successfully saved to {OUTLIERS_PATH}')
        except Exception as e:
            self.logger.error(f'An error occured during JSON operation: {e}')
            raise
        mlflow.log_artifact(OUTLIERS_PATH)        