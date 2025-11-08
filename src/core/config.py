"""
Configuration Management Module
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict

class Config:
    """Configuration manager for VSim"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file"""
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        paths = self.config.get('paths', {})
        for key, path in paths.items():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'genome_analysis.min_orf_length')"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config"""
        return key in self.config

