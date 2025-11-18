"""Configuration management for Store Item Detection."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for managing project settings."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path is None:
            return self._get_default_config()
        
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'model': {
                'name': 'yolov8',
                'input_size': 640,
                'num_classes': 10,
                'pretrained': True
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'early_stopping_patience': 10
            },
            'data': {
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'image_size': [640, 640],
                'augmentation': True
            },
            'paths': {
                'data_dir': 'data',
                'model_dir': 'models',
                'output_dir': 'outputs',
                'log_dir': 'logs'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any):
        """Update configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
