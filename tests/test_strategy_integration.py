"""
Integration tests for running strategies with the configuration system.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config_loader import get_strategy_params
from src.strategies.equal_weight import run_equal_weight_backtest
from src.strategies.momentum_strategy import run_momentum_backtest
from src.strategies.momentum_hrp import run_momentum_hrp_backtest


@pytest.fixture
def test_returns():
    """Load test return data."""
    test_data_path = Path(__file__).parent / 'data' / 'test_returns.csv'
    returns = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
    return returns


@pytest.fixture
def test_config_path():
    """Path to test configuration file."""
    return Path(__file__).parent / 'config' / 'test_config.yaml'


def test_equal_weight_strategy(test_returns, test_config_path):
    """Test running the equal weight strategy using the configuration."""
    # Get strategy parameters from the test config
    params = get_strategy_params('equal_weight', config_file=test_config_path)
    
    # Run the strategy
    portfolio_values, weights = run_equal_weight_backtest(
        test_returns,
        lookback_days=params['lookback_days'],
        initial_capital=params['initial_capital'],
        rebalancing_freq=params['rebalance_frequency'],
        n_assets=params.get('n_assets')
    )
    
    # Verify we get results
    assert len(portfolio_values) > 0
    assert portfolio_values.iloc[0] == params['initial_capital']
    
    # Check that weights are calculated as expected
    assert len(weights) > 0
    # If n_assets is specified, verify we're using at most that many assets
    if 'n_assets' in params and params['n_assets'] is not None:
        # Check any non-zero row of weights
        non_zero_dates = weights.sum(axis=1)
        non_zero_dates = non_zero_dates[non_zero_dates > 0]
        
        if len(non_zero_dates) > 0:
            date_to_check = non_zero_dates.index[0]
            active_positions = weights.loc[date_to_check][weights.loc[date_to_check] > 0]
            assert len(active_positions) <= params['n_assets']


def test_momentum_strategy(test_returns, test_config_path):
    """Test running the momentum strategy using the configuration."""
    # Get strategy parameters from the test config
    params = get_strategy_params('momentum', config_file=test_config_path)
    
    # Run the strategy
    portfolio_values, weights = run_momentum_backtest(
        test_returns,
        lookback_days=params['lookback_days'],
        momentum_window=params['momentum_window'],
        initial_capital=params['initial_capital'],
        rebalancing_freq=params['rebalance_frequency'],
        n_assets=params.get('n_assets')
    )
    
    # Verify we get results
    assert len(portfolio_values) > 0
    assert abs(portfolio_values.iloc[0] - params['initial_capital']) < 1e-10
    
    # Check active positions match n_assets
    if 'n_assets' in params and params['n_assets'] is not None:
        non_zero_dates = weights.sum(axis=1)
        non_zero_dates = non_zero_dates[non_zero_dates > 0]
        
        if len(non_zero_dates) > 0:
            date_to_check = non_zero_dates.index[0]
            active_positions = weights.loc[date_to_check][weights.loc[date_to_check] > 0]
            assert len(active_positions) <= params['n_assets']


def test_momentum_hrp_strategy(test_returns, test_config_path):
    """Test running the momentum-HRP hybrid strategy using the configuration."""
    # Get strategy parameters from the test config
    params = get_strategy_params('momentum_hrp', config_file=test_config_path)
    
    # Run the strategy
    portfolio_values, weights = run_momentum_hrp_backtest(
        test_returns,
        lookback_days=params['lookback_days'],
        momentum_window=params['momentum_window'],
        initial_capital=params['initial_capital'],
        rebalancing_freq=params['rebalance_frequency'],
        n_momentum_assets=params.get('n_momentum_assets')
    )
    
    # Verify we get results
    assert len(portfolio_values) > 0
    assert abs(portfolio_values.iloc[0] - params['initial_capital']) < 1e-10
    
    # Check that weights are calculated
    assert len(weights) > 0
