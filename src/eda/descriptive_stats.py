"""
Descriptive statistics computation with optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from utils import read_yaml, LoggerMixin, ensure_directory, write_json
import mlflow

class DescriptiveStats(LoggerMixin):
    '''Performs preliminary data exploration'''

    def __init__(self, config):
        self.config = config
        self.logger = self.setup_class_logger('Descriptive_stats', config,'logging')

    def compute_basic_stats(self,df: pd.DataFrame) -> Dict[str, any]:
        """
        Compute basic statistics efficiently.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with basic statistics
        """
        self.logger.info("Computing basic descriptive statistics...")
        
        stats = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'columns': df.columns.tolist()
        }

        
        self.logger.info(f"Dataset: {stats['n_rows']:,} rows x {stats['n_columns']} columns")
        self.logger.info(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
        
        return stats


    def analyze_numeric_columns(self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Analyze numeric columns with summary statistics.
        
        Args:
            df: Input DataFrame
            id_column: ID column to exclude from analysis
            
        Returns:
            Tuple of (summary DataFrame, list of numeric columns)
        """
        self.logger.info("Analyzing numeric columns...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        id_column: Optional[str] = self.config['data'].get('id_column', None)
        
        # Exclude ID column
        if id_column  in numeric_cols:
            numeric_cols.remove(id_column)
        
        if not numeric_cols:
            self.logger.warning("No numeric columns found")
            return pd.DataFrame(), []
        
        # Compute statistics (vectorized)
        summary = df[numeric_cols].describe().T
        summary['range'] = summary['max'] - summary['min']
        summary['cv'] = (summary['std'] / summary['mean']).round(4)  # Coefficient of variation
        
        self.logger.info(f"Analyzed {len(numeric_cols)} numeric columns")
        
        return summary, numeric_cols


    def analyze_categorical_columns(self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Analyze categorical columns.
        
        Args:
            df: Input DataFrame
            target_column: Target column (will be included in analysis)
            
        Returns:
            Tuple of (summary DataFrame, list of categorical columns)
        """
        self.logger.info("Analyzing categorical columns...")

        target_column: Optional[str] = self.config['data'].get('target_column', None)      

        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if not cat_cols:
            self.logger.warning("No categorical columns found")
            return pd.DataFrame(), []
        
        summary_data = []
        for col in cat_cols:
            summary_data.append({
                'column': col,
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
                'top_5_values': df[col].value_counts().head(5).to_dict()
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        self.logger.info(f"Analyzed {len(cat_cols)} categorical columns")
        
        return summary_df, cat_cols
    
                # Target variable distribution
    def target_col_dist(self, df):
        '''Analyze the distribution of the target variable'''

        target_col = self.config['data'].get('target_column','Loan Status')   

        try:
            if target_col and target_col in df.columns:
                target_dist = df[target_col].value_counts().to_dict()

                        
                self.logger.info(f"Target variable '{target_col}' distribution:")
                for value, count in target_dist.items():
                    pct = (count / len(self.df)) * 100
                    self.logger.info(f"  {value}: {count:,} ({pct:.2f}%)")
                
        except Exception as e:
            self.logger.error(f"Target analysis failed: {e}", exc_info=True)
            raise 
    
    
    def run_descriptive_stats(self, df):
        '''Run all descriptive statistics'''
        # BASIC STATISTICS
        stats = self.compute_basic_stats(df)

        # log basic stats
        mlflow.log_metric('n_rows',stats['n_rows'])
        mlflow.log_metric('n_columns',stats['n_columns'])
        mlflow.log_metric('memory_usage_mb',stats['memory_usage_mb'])

        STATS_PATH = Path(f'data/artifacts/basic_stats.json')

        try:
            self.logger.info(f'Writing basic descriptive stats results to a json file..')
            write_json(stats, STATS_PATH, indent=4)
            self.logger.info(f'Data written succcessfully written to: {STATS_PATH}')
            mlflow.log_artifact(STATS_PATH)
            self.logger.info(f'Descriptive stats results logged to MLflow successfully')
        except Exception as e:
            self.logger.error(f'An error was encountered during this operation: {e}')
            raise

        # # analyzing numeric columns
        summary, numeric_cols = self.analyze_numeric_columns(df)

        SUMMARY_PATH = Path(f'data/artifacts/summary_path.csv')
        try:
            self.logger.info(f"Writing data to csv file...")
            summary.to_csv(SUMMARY_PATH)
            self.logger.info(f'Summary data written successfully to: {SUMMARY_PATH}')
        except FileNotFoundError as e:
            self.logger.error(f'An error occured because file path was not found: {e}')
            raise
        except Exception as e:
            self.logger.error(f'Error saving summary to {SUMMARY_PATH}: {e}')
            raise
        mlflow.log_artifact(SUMMARY_PATH)


        # analyzing categorical columns
        summary_df, cat_cols = self.analyze_categorical_columns(df)

        SUMMARY_PATH = Path(f'data/artifacts/categorical_summary.csv')

        try:
            self.logger.info(f"Writing data to csv file...")
            summary_df.to_csv(SUMMARY_PATH)
            self.logger.info(f'Summary data written successfully to: {SUMMARY_PATH}')
        except FileNotFoundError as e:
            self.logger.error(f'An error occured because file path was not found: {e}')
            raise
        except Exception as e:
            self.logger.error(f'Error saving summary to {SUMMARY_PATH}: {e}')
            raise
        mlflow.log_artifact(SUMMARY_PATH)

        features = numeric_cols + cat_cols
        feature_store = Path('data/artifacts/feature_store.json')


        try:
            self.logger.info(f'Saving feature names to a json file...')
            write_json(features, feature_store, indent=4)
            self.logger.info(f'Data successfully saved to : {feature_store}')
        except Exception as e:
            self.logger.error(f'An error occured while handling json operation: {e}')
            raise
        mlflow.log_artifact(feature_store)



