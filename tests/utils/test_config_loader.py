"""
Tests for the configuration loader utility.
"""

import os
import pytest
from pathlib import Path
import tempfile
import yaml
from src.utils.config_loader import load_config, get_strategy_params


def test_load_config_file_not_found():
    """Test that load_config raises FileNotFoundError when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_config(config_file='non_existent_file.yaml')


def test_load_config_valid_file():
    """Test that load_config correctly loads configuration from a valid file."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp:
        config_content = """
        general:
          initial_capital: 50000
        hrp:
          lookback_days: 30
        """
        temp.write(config_content.encode('utf-8'))
        temp_path = temp.name
    
    try:
        # Test loading the config
        config = load_config(config_file=temp_path)
        
        # Verify the config was loaded correctly
        assert 'general' in config
        assert config['general']['initial_capital'] == 50000
        assert 'hrp' in config
        assert config['hrp']['lookback_days'] == 30
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)


def test_get_strategy_params():
    """Test that get_strategy_params correctly merges general and strategy-specific parameters."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp:
        config_content = """
        general:
          initial_capital: 100000
          rebalance_frequency: 'W'
          data_file: 'data/daily_returns.csv'
        
        hrp:
          lookback_days: 60
        
        momentum:
          lookback_days: 50
          momentum_window: 30
        """
        temp.write(config_content.encode('utf-8'))
        temp_path = temp.name
    
    try:
        # Test getting params for hrp strategy
        hrp_params = get_strategy_params('hrp', config_file=temp_path)
        
        # Verify the params were merged correctly
        assert hrp_params['initial_capital'] == 100000
        assert hrp_params['rebalance_frequency'] == 'W'
        assert hrp_params['lookback_days'] == 60
        
        # Test getting params for momentum strategy
        momentum_params = get_strategy_params('momentum', config_file=temp_path)
        
        # Verify the params were merged correctly
        assert momentum_params['initial_capital'] == 100000
        assert momentum_params['lookback_days'] == 50
        assert momentum_params['momentum_window'] == 30
        
        # Test getting params for a non-existent strategy (should return general params)
        missing_strategy_params = get_strategy_params('missing', config_file=temp_path)
        assert missing_strategy_params['initial_capital'] == 100000
        assert 'lookback_days' not in missing_strategy_params
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)
