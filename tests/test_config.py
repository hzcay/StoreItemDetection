"""Tests for configuration module."""

import pytest
import tempfile
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from store_detection.config import Config


def test_default_config():
    """Test default configuration."""
    config = Config()
    
    assert config.get('model.name') == 'yolov8'
    assert config.get('model.input_size') == 640
    assert config.get('training.batch_size') == 16


def test_load_config_from_file():
    """Test loading configuration from file."""
    # Create temporary config file
    config_data = {
        'model': {
            'name': 'test_model',
            'input_size': 512
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        config = Config(config_path)
        assert config.get('model.name') == 'test_model'
        assert config.get('model.input_size') == 512
    finally:
        Path(config_path).unlink()


def test_config_get():
    """Test getting configuration values."""
    config = Config()
    
    # Test nested key
    assert config.get('model.name') is not None
    
    # Test default value
    assert config.get('nonexistent.key', 'default') == 'default'


def test_config_update():
    """Test updating configuration values."""
    config = Config()
    
    config.update('model.name', 'new_model')
    assert config.get('model.name') == 'new_model'
    
    config.update('new.nested.key', 'value')
    assert config.get('new.nested.key') == 'value'


def test_config_save():
    """Test saving configuration."""
    config = Config()
    config.update('model.name', 'saved_model')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        save_path = f.name
    
    try:
        config.save(save_path)
        
        # Load saved config
        loaded_config = Config(save_path)
        assert loaded_config.get('model.name') == 'saved_model'
    finally:
        Path(save_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__])
