"""
Configuration Loader Utility

This module provides utilities for loading configuration parameters from YAML files.
"""

import yaml
from pathlib import Path


def load_config(config_file=None):
    """
    Load configuration parameters from a YAML file.
    
    Parameters
    ----------
    config_file : str or Path, optional
        Path to the configuration file. If None, uses default config location.
        
    Returns
    -------
    dict
        Dictionary containing configuration parameters.
    """
    if config_file is None:
        # Default config file location
        config_file = Path(__file__).parent.parent.parent / 'config' / 'strategy_params.yaml'
    else:
        config_file = Path(config_file)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_strategy_params(strategy_name, config_file=None):
    """
    Get parameters for a specific strategy.
    
    Parameters
    ----------
    strategy_name : str
        Name of the strategy (e.g., 'hrp', 'momentum', etc.)
    config_file : str or Path, optional
        Path to the configuration file. If None, uses default config location.
        
    Returns
    -------
    dict
        Dictionary containing strategy-specific parameters.
    """
    config = load_config(config_file)
    
    # Get general parameters (default to empty dict if not found or if value is None)
    general_params = config.get('general', {}) or {}
    
    # Get strategy-specific parameters (default to empty dict if not found or if value is None)
    strategy_params = config.get(strategy_name, {}) or {}
    
    # No need for extra None checks since we've handled it above
    # using the `or {}` pattern which returns {} when the left side evaluates to None
    
    # Merge general and strategy-specific parameters
    # Strategy-specific parameters take precedence
    combined_params = {**general_params, **strategy_params}
    
    return combined_params
