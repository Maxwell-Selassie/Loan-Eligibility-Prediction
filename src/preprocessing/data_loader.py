"""
Data loading module with validation and optimization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import read_csv, Timer

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and prepare data for preprocessing pipeline.
    
    Attributes:
        config: Configuration dictionary
        df: Loaded DataFrame
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        
    def load(self) -> pd.DataFrame:
        """
        Load raw data from file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data is empty or corrupted
        """
        with Timer("Data loading", logger):
            file_path = self.config['file_paths']['raw_data']
            
            logger.info(f"Loading raw data from: {file_path}")
            
            try:
                self.df = read_csv(
                    filepath=file_path,
                    optimize_dtypes=True
                )
                
                if self.df.empty:
                    raise ValueError("Loaded data is empty")
                
                logger.info(f"Data loaded successfully: {len(self.df):,} rows × {len(self.df.columns)} columns")
                logger.info(f"Columns: {self.df.columns.tolist()}")
                logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                return self.df
                
            except Exception as e:
                logger.error(f"Failed to load data: {e}", exc_info=True)
                raise
    
    def get_data(self) -> pd.DataFrame:
        """Get loaded DataFrame."""
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self.df
