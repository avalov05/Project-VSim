"""
Logging Setup Module
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(config, level: int = logging.INFO) -> logging.Logger:
    """Setup and configure logger"""
    logger = logging.getLogger('VSim')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # File handler
    log_config = config.get('logging', {})
    log_file = log_config.get('file', 'logs/vsim.log')
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    
    # Formatter
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

