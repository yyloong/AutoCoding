#!/usr/bin/env python3
"""
Configuration management module.
Handles loading and providing access to application configuration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration class to manage application settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration.
        
        Args:
            config_path (str, optional): Path to configuration file. 
                                        Defaults to None (uses default config).
        """
        self._config = {}
        self._load_default_config()
        self._load_config_from_file(config_path)
        
    def _load_default_config(self):
        """Load default configuration values."""
        self._config = {
            "app_name": "MyApplication",
            "version": "0.1.0",
            "debug": False,
            "log_level": "INFO"
        }
        
    def _load_config_from_file(self, config_path: Optional[str] = None):
        """
        Load configuration from a file if provided.
        
        Args:
            config_path (str, optional): Path to configuration file.
        """
        if config_path is None:
            # Try to find config in common locations
            config_locations = [
                "./config.json",
                "../config.json",
                "/etc/myapp/config.json"
            ]
            
            for location in config_locations:
                if os.path.exists(location):
                    config_path = location
                    break
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    self._config.update(file_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load configuration from {config_path}: {e}")
        else:
            logger.info("No external configuration file found, using defaults")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            dict: Complete configuration dictionary.
        """
        return self._config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            key (str): Configuration key.
            default (Any): Default value if key is not found.
            
        Returns:
            Any: Configuration value or default.
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key (str): Configuration key.
            value (Any): Configuration value.
        """
        self._config[key] = value
        logger.debug(f"Set configuration key '{key}' to '{value}'")
    
    def reload(self, config_path: Optional[str] = None):
        """
        Reload configuration from file.
        
        Args:
            config_path (str, optional): Path to configuration file.
        """
        self._load_default_config()
        self._load_config_from_file(config_path)
        logger.info("Configuration reloaded")