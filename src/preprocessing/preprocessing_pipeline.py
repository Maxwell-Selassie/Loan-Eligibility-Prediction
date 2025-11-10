"""
Main Preprocessing Pipeline Orchestrator
Production-grade with comprehensive error handling, logging, and validation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    read_yaml, write_csv, write_json, ensure_directory,
    get_timestamp, Timer, setup_logger
)
from preprocessing import (
    DataLoader, DataValidator, FeatureDropper,
    DataEncoder, DataScaler, DataSplitter
)


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


class PreprocessingPipeline:
    """
    Production-grade preprocessing pipeline for Loan Eligibility Prediction.
    
    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        timestamp: Pipeline execution timestamp
        metadata: Dictionary storing preprocessing metadata
    """
    
    def __init__(self, config_path: str = "config/preprocessing_config.yaml"):
        """
        Initialize Preprocessing Pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.timestamp = get_timestamp()
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.metadata: Dict[str, Any] = {
            'timestamp': self.timestamp,
            'config_path': config_path
        }
        
        # Initialize components
        self.loader: Optional[DataLoader] = None
        self.validator: Optional[DataValidator] = None
        self.dropper: Optional[FeatureDropper] = None
        self.encoder: Optional[DataEncoder] = None
        self.scaler: Optional[DataScaler] = None
        self.splitter: Optional[DataSplitter] = None
        
        # Data storage
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_val: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        
        self.logger.info("="*80)
        self.logger.info(f"PREPROCESSING PIPELINE INITIALIZED - {self.timestamp}")
        self.logger.info("="*80)
        self.logger.info(f"Author: {self.config['project']['name']}")
        self.logger.info(f"Version: {self.config['project']['version']}")
        self.logger.info(f"Description: {self.config['project']['description']}")
    
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
        log_dir = Path(log_config.get('log_dir', 'logs/'))
        
        ensure_directory(log_dir)
        
        logger = setup_logger(
            name='preprocessing_pipeline',
            log_dir=log_dir,
            log_level=log_config.get('log_level', 'INFO'),
            max_bytes=log_config.get('max_bytes', 10485760),
            backup_count=log_config.get('backup_count', 7)
        )
        
        return logger
    
    def _ensure_output_directories(self) -> None:
        """Ensure all output directories exist."""
        self.logger.info("Creating output directories...")
        
        file_paths = self.config.get('file_paths', {})
        
        # Ensure models directory
        scaler_path = Path(file_paths.get('scaler_path', 'models/scaler.pkl'))
        ensure_directory(scaler_path.parent)
        
        # Ensure processed data directory
        train_path = Path(file_paths.get('train_preprocessed_data', 'data/processed/train.csv'))
        ensure_directory(train_path.parent)
        
        # Ensure artifacts directory
        metadata_path = Path(file_paths.get('preprocessing_metadata', 'data/artifacts/metadata.json'))
        ensure_directory(metadata_path.parent)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data.
        
        Returns:
            Loaded DataFrame
        """
        with Timer("Data loading", self.logger):
            try:
                self.loader = DataLoader(self.config)
                self.df_raw = self.loader.load()
                
                # Store metadata
                self.metadata['raw_data'] = {
                    'n_rows': len(self.df_raw),
                    'n_columns': len(self.df_raw.columns),
                    'columns': self.df_raw.columns.tolist(),
                    'memory_mb': self.df_raw.memory_usage(deep=True).sum() / 1024**2
                }
                
                return self.df_raw
                
            except Exception as e:
                self.logger.error(f"Data loading failed: {e}", exc_info=True)
                raise PreprocessingError(f"Failed to load data: {e}")
    
    def validate_data(self, df: pd.DataFrame, stage: str = 'raw') -> pd.DataFrame:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            stage: Pipeline stage
            
        Returns:
            Validated DataFrame
        """
        with Timer(f"Data validation ({stage})", self.logger):
            try:
                if self.validator is None:
                    self.validator = DataValidator(self.config)
                
                df_validated = self.validator.validate_all(df, stage=stage)
                
                return df_validated
                
            except Exception as e:
                self.logger.error(f"Data validation failed: {e}", exc_info=True)
                raise PreprocessingError(f"Validation failed: {e}")
    
    def drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop unnecessary features.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with dropped features
        """
        with Timer("Feature dropping", self.logger):
            try:
                self.dropper = FeatureDropper(self.config)
                df_dropped = self.dropper.fit_transform(df)
                
                # Store metadata
                self.metadata['dropped_columns'] = self.dropper.dropped_columns
                
                return df_dropped
                
            except Exception as e:
                self.logger.error(f"Feature dropping failed: {e}", exc_info=True)
                raise PreprocessingError(f"Feature dropping failed: {e}")
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets.
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        with Timer("Data splitting", self.logger):
            try:
                target_col = self.config['file_paths'].get('target_column', 'Loan_Status')
                
                # Handle target column from label encoding config
                if not target_col or target_col == 'Loan_Status':
                    label_config = self.config.get('encoding', {}).get('label_encode', {})
                    if label_config:
                        target_col = list(label_config.keys())[0]
                
                self.splitter = DataSplitter(self.config)
                df_train, df_test = self.splitter.split(df, target_col)
                
                # Store metadata
                self.metadata['data_split'] = {
                    'target_column': target_col,
                    'train_size': len(df_train),
                    'test_size': len(df_test),
                    'train_indices': self.splitter.split_indices['train'].tolist(),
                    'test_indices': self.splitter.split_indices['test'].tolist()
                }
                
                return df_train, df_test
                
            except Exception as e:
                self.logger.error(f"Data splitting failed: {e}", exc_info=True)
                raise PreprocessingError(f"Data splitting failed: {e}")
    
    def encode_features(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features.
        
        Args:
            df_train: Training DataFrame
            df_val: Validation DataFrame
            df_test: Test DataFrame
            
        Returns:
            Tuple of encoded DataFrames
        """
        with Timer("Feature encoding", self.logger):
            try:
                self.encoder = DataEncoder(self.config)
                
                # Fit on train, transform all
                df_train_encoded = self.encoder.fit_transform(df_train)
                df_test_encoded = self.encoder.transform(df_test)
                
                # Align columns (in case of missing categories in val/test)
                all_columns = df_train_encoded.columns.tolist()
                
                for col in all_columns:
                    if col not in df_test_encoded.columns:
                        df_test_encoded[col] = 0
                
                df_test_encoded = df_test_encoded[all_columns]
                
                # Store metadata
                self.metadata['encoding'] = {
                    'encoding_mappings': self.encoder.encoding_mappings,
                    'encoded_columns': self.encoder.encoded_columns
                }
                
                # Save encoder
                encoder_path = self.config['file_paths'].get('encoder_path', 'models/encoder.pkl')
                self.encoder.save(encoder_path)
                
                return df_train_encoded, df_test_encoded
                
            except Exception as e:
                self.logger.error(f"Feature encoding failed: {e}", exc_info=True)
                raise PreprocessingError(f"Feature encoding failed: {e}")
    
    def scale_features(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale numeric features.
        
        Args:
            df_train: Training DataFrame
            df_val: Validation DataFrame
            df_test: Test DataFrame
            
        Returns:
            Tuple of scaled DataFrames
        """
        with Timer("Feature scaling", self.logger):
            try:
                self.scaler = DataScaler(self.config)
                
                # Fit on train, transform all
                df_train_scaled = self.scaler.fit_transform(df_train)
                df_test_scaled = self.scaler.transform(df_test)
                
                # Store metadata
                if self.scaler.fitted:
                    self.metadata['scaling'] = {
                        'method': self.config['scaling'].get('method', 'standard'),
                        'columns_scaled': self.scaler.columns_to_scale
                    }
                    
                    # Add scaler parameters
                    if hasattr(self.scaler.scaler, 'mean_'):
                        self.metadata['scaling']['parameters'] = {
                            'mean': self.scaler.scaler.mean_.tolist(),
                            'scale': self.scaler.scaler.scale_.tolist()
                        }
                
                # Save scaler
                scaler_path = self.config['file_paths'].get('scaler_path', 'models/scaler.pkl')
                self.scaler.save(scaler_path)
                
                return df_train_scaled, df_test_scaled
                
            except Exception as e:
                self.logger.error(f"Feature scaling failed: {e}", exc_info=True)
                raise PreprocessingError(f"Feature scaling failed: {e}")
    
    def check_distribution_shift(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> None:
        """
        Check for distribution shifts between splits.
        
        Args:
            df_train: Training DataFrame
            df_val: Validation DataFrame
            df_test: Test DataFrame
        """
        if self.validator is None:
            return
        
        with Timer("Distribution shift detection", self.logger):
            try:
                # Get numeric columns
                numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
                
                # Remove target if present
                label_config = self.config.get('encoding', {}).get('label_encode', {})
                if label_config:
                    target_col = list(label_config.keys())[0]
                    if target_col in numeric_cols:
                        numeric_cols.remove(target_col)
                
                # Check train vs val
                shift_train_val = self.validator.check_distribution_shift(
                    df_train, numeric_cols, 'validation'
                )
                
                # Check train vs test
                shift_train_test = self.validator.check_distribution_shift(
                    df_train, df_test, numeric_cols, 'test'
                )
                
                # Store in metadata
                self.metadata['distribution_shift'] = {
                    'train_vs_test': shift_train_test
                }
                
            except Exception as e:
                self.logger.warning(f"Distribution shift check failed: {e}")
    
    def save_processed_data(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> None:
        """
        Save processed datasets.
        
        Args:
            df_train: Training DataFrame
            df_val: Validation DataFrame
            df_test: Test DataFrame
        """
        with Timer("Saving processed data", self.logger):
            try:
                file_paths = self.config['file_paths']
                
                # Save train
                train_path = file_paths.get('train_preprocessed_data')
                write_csv(df_train, train_path)
                self.logger.info(f"Train set saved: {train_path} ({len(df_train):,} rows)")
                
                
                # Save test
                test_path = file_paths.get('test_preprocessed_data')
                write_csv(df_test, test_path)
                self.logger.info(f"Test set saved: {test_path} ({len(df_test):,} rows)")
                
            except Exception as e:
                self.logger.error(f"Failed to save processed data: {e}", exc_info=True)
                raise PreprocessingError(f"Failed to save data: {e}")
    
    def save_metadata(self) -> None:
        """Save preprocessing metadata."""
        with Timer("Saving metadata", self.logger):
            try:
                metadata_path = self.config['file_paths'].get('preprocessing_metadata')
                
                # Add final statistics
                self.metadata['final_statistics'] = {
                    'train_shape': (len(self.df_train), len(self.df_train.columns)) if self.df_train is not None else None,
                    'test_shape': (len(self.df_test), len(self.df_test.columns)) if self.df_test is not None else None,
                }
                
                write_json(self.metadata, metadata_path)
                self.logger.info(f"Metadata saved: {metadata_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to save metadata: {e}", exc_info=True)
    
    def execute(self) -> Dict[str, pd.DataFrame]:
        """
        Execute complete preprocessing pipeline.
        
        Returns:
            Dictionary with processed datasets
        """
        try:
            with Timer("Complete Preprocessing Pipeline", self.logger):
                # Setup
                self._ensure_output_directories()
                
                # Stage 1: Load data
                df = self.load_data()
                
                # Stage 2: Validate raw data
                df = self.validate_data(df, stage='raw')
                
                # Stage 3: Drop unnecessary features
                df = self.drop_features(df)
                
                # Stage 4: Split data (before encoding/scaling)
                df_train, df_test = self.split_data(df)
                
                # Stage 5: Encode features (fit on train)
                df_train, df_test = self.encode_features(df_train, df_test)
                
                # Stage 6: Scale features (fit on train)
                df_train, df_test = self.scale_features(df_train, df_test)
                
                # Stage 7: Check distribution shift
                self.check_distribution_shift(df_train, df_test)
                
                # Stage 8: Save processed data
                self.df_train = df_train
                self.df_test = df_test
                
                self.save_processed_data(df_train, df_test)
                
                # Stage 9: Save metadata
                self.save_metadata()
                
                self.logger.info("="*80)
                self.logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
                self.logger.info("="*80)
                self.logger.info(f"Train set: {len(df_train):,} rows × {len(df_train.columns)} columns")
                self.logger.info(f"Test set: {len(df_test):,} rows × {len(df_test.columns)} columns")
                
                return {
                    'train': df_train,
                    'test': df_test
                }
                
        except PreprocessingError:
            raise
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise PreprocessingError(f"Pipeline failed: {e}")


def main():
    """Main entry point for preprocessing pipeline."""
    try:
        # Initialize pipeline
        pipeline = PreprocessingPipeline(config_path="config/preprocessing_config.yaml")
        
        # Execute
        datasets = pipeline.execute()
        
        return 0
        
    except PreprocessingError as e:
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

