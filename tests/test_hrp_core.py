"""
Tests for the core HRP functionality
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Import the functions to test
from src.utils.hrp_utils import corr_distance, get_cluster_var, extract_clusters, get_hrp_weights
from src.strategies.hrp import run_hrp_backtest


@pytest.fixture
def sample_returns():
    """Generate a sample returns DataFrame for testing."""
    # Load from test data
    test_data_path = Path(__file__).parent / 'data' / 'test_returns.csv'
    returns = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
    return returns


def test_corr_distance():
    """Test the correlation distance function."""
    # Create a test correlation matrix
    corr = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ])
    
    # Calculate the distance matrix
    dist = corr_distance(corr)
    
    # Expected: distance based on sqrt((1 - corr) / 2)
    expected = np.sqrt((1 - corr) / 2)
    
    np.testing.assert_almost_equal(dist, expected)


def test_get_cluster_var(sample_returns):
    """Test the cluster variance calculation function."""
    # Use the first 3 assets from the sample returns
    returns_subset = sample_returns.iloc[:, :3]
    
    # Calculate covariance matrix
    cov = returns_subset.cov()
    
    # Create a simple cluster using column names instead of indices
    cluster = list(returns_subset.columns)  # Use actual column names
    
    # Calculate cluster variance
    cluster_var = get_cluster_var(cov, cluster)
    
    # Verify the result is a positive number
    assert cluster_var > 0


def test_extract_clusters():
    """Test the cluster extraction function."""
    # Create a sample linkage matrix
    # Format: [idx1, idx2, distance, count]
    link = np.array([
        [0, 1, 0.5, 2],
        [2, 3, 0.6, 2],
        [4, 5, 0.7, 2],
        [6, 7, 0.3, 4]
    ])
    
    # Create sample labels
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    # Extract clusters
    clusters = extract_clusters(link, labels)
    
    # Verify we get the expected number of clusters
    assert len(clusters) > 0


def test_get_hrp_weights(sample_returns):
    """Test the HRP weights calculation function."""
    # Use the first 5 assets from the sample returns
    returns_subset = sample_returns.iloc[:, :3]
    
    # Calculate correlation matrix
    corr = returns_subset.corr()
    
    # Calculate weights using HRP
    weights = get_hrp_weights(corr, returns_subset.cov())
    
    # Check that weights sum to approximately 1
    assert abs(sum(weights) - 1.0) < 1e-10
    
    # Check that all weights are positive
    assert all(w > 0 for w in weights)


def test_run_hrp_backtest(sample_returns):
    """Integration test for the HRP backtest function."""
    # Run a simple backtest
    portfolio_values, weights_history = run_hrp_backtest(
        sample_returns,
        lookback_days=5,
        rebalancing_freq='W',
        initial_capital=10000
    )
    
    # Check that we have portfolio values for each date
    assert len(portfolio_values) > 0
    
    # Check that the first portfolio value equals the initial capital
    assert abs(portfolio_values.iloc[0] - 10000) < 1e-10
    
    # Check that we have weights for at least one date
    assert len(weights_history) > 0
