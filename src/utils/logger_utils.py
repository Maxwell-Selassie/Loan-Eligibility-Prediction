"""
Production-grade logging utilities with security and performance optimizations.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Union
import os


def setup_logger(
    name: str,
    log_dir: Union[str, Path],
    log_level: str = "INFO",
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 7,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    console_output: bool = True,
    file_permissions: int = 0o640
) -> logging.Logger:
    """
    Setup production-grade logger with file rotation and console output.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        log_format: Custom log format string
        date_format: Custom date format string
        console_output: Whether to output to console
        file_permissions: Unix file permissions for log files
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger("eda_pipeline", "../logs/", log_level="INFO")
        >>> logger.info("Pipeline started")
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Prevent adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Set log level
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level_obj)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Default formats
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # File handler with rotation
    log_file = log_dir / f"{name}.log"
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level_obj)
    logger.addHandler(file_handler)
    
    # Set file permissions (Unix only)
    if hasattr(os, 'chmod'):
        try:
            os.chmod(log_file, file_permissions)
        except Exception:
            pass  # Fail silently if permissions can't be set
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level_obj)
        logger.addHandler(console_handler)
    
    logger.info(f"Logger '{name}' initialized with level {log_level}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)