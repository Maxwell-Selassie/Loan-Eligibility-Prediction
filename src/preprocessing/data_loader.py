"""
Data loading module with validation and optimization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import mlflow

# Add parent directory to path
from utils import read_csv, Timer, LoggerMixin




class DataLoader(LoggerMixin):
    """
    Load and prepare data for preprocessing pipeline. 
    This class loads raw data from the source (csv file path)
    that will be processsed for model training
    
    Attributes:
        config: Configuration dictionary
        df: Loaded DataFrame

    Examples:
        >>> load_data = DataLoader(config)
        >>> load_data.load()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataLoader. 
        Configurations are in a yaml file(config/preprocessing_config.yaml)
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = self.setup_class_logger('data_loader', config, 'logging')
        
    def load(self) -> pd.DataFrame:
        """
        Load raw data from file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data is empty or corrupted
        """
        with Timer("Data loading", self.logger):
            file_path = self.config['file_paths']['raw_data']
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f'File Not Found! Check filepath and try again!')
                raise FileNotFoundError(f'File Not Found!')
            
            self.logger.info(f"Loading raw data from: {file_path}")
            
            try:
                df = read_csv(
                    filepath=file_path,
                    optimize_dtypes=True
                )
                
                if self.df.empty:
                    raise ValueError("Loaded data is empty")
                
                self.logger.info(f"Data loaded successfully: {len(self.df):,} rows × {len(self.df.columns)} columns")
                self.logger.info(f"Columns: {self.df.columns.tolist()}")
                self.logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

                mlflow.log_param('no_of_rows', len(self.df))
                mlflow.log_param('no_of_features', len(self.df.columns))
                
                return df
                
            except Exception as e:
                self.logger.error(f"Failed to load data: {e}", exc_info=True)
                raise
    
