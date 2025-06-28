"""
Data Loading Utilities

This module contains functions for loading and preprocessing financial data
for portfolio strategies, including symbol filtering and data validation.
"""

import pandas as pd
from pathlib import Path
from .config_loader import load_config


def get_symbols_from_config(config_path=None):
    """
    Get list of symbols from configuration file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file
        
    Returns
    -------
    list or None
        List of symbols to use, or None to use all from data
    """
    config = load_config(config_path)
    
    if 'symbols' in config:
        symbols_config = config['symbols']
        
        # If use_all_from_data is True, return None to indicate all symbols should be used
        if symbols_config.get('use_all_from_data', False):
            return None
        
        # Otherwise return the specified crypto symbols
        return symbols_config.get('crypto_symbols', [])
    
    return None


def load_returns_data(data_file, config_path=None, symbols=None):
    """
    Load returns data from CSV file with optional symbol filtering.
    
    Parameters
    ----------
    data_file : str
        Path to the CSV file containing returns data
    config_path : str, optional
        Path to configuration file for symbol filtering
    symbols : list, optional
        Explicit list of symbols to filter (overrides config)
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing daily returns for selected symbols
    """
    print(f"Loading returns data from {data_file}")
    
    # Check if file exists
    if not Path(data_file).exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load the data
    try:
        returns = pd.read_csv(data_file, index_col=0, parse_dates=True)
    except Exception as e:
        raise ValueError(f"Error loading data from {data_file}: {e}")
    
    # Get symbols to filter if not explicitly provided
    if symbols is None:
        symbols = get_symbols_from_config(config_path)
    
    # Filter symbols if specified
    if symbols is not None:
        # Find available symbols that match the requested ones
        available_symbols = [sym for sym in symbols if sym in returns.columns]
        
        if not available_symbols:
            print(f"Warning: None of the specified symbols {symbols} found in data")
            print(f"Available symbols: {list(returns.columns)}")
            print("Using all available symbols")
        else:
            print(f"Filtering to {len(available_symbols)} symbols: {available_symbols}")
            returns = returns[available_symbols]
    
    # Sort by date
    returns = returns.sort_index()
    
    # Basic validation
    if returns.empty:
        raise ValueError("No data loaded or all data filtered out")
    
    if len(returns.columns) < 2:
        raise ValueError("Need at least 2 assets for portfolio strategies")
    
    print(f"Loaded data: {len(returns)} rows, {len(returns.columns)} assets")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    return returns


def validate_returns_data(returns):
    """
    Validate returns data quality and provide warnings.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data to validate
        
    Returns
    -------
    dict
        Dictionary containing validation results and warnings
    """
    validation_results = {
        'warnings': [],
        'stats': {}
    }
    
    # Check for missing data
    missing_data = returns.isnull().sum()
    if missing_data.sum() > 0:
        validation_results['warnings'].append(f"Missing data found in {(missing_data > 0).sum()} symbols")
        validation_results['stats']['missing_data_by_symbol'] = missing_data[missing_data > 0].to_dict()
    
    # Check for extreme values
    for col in returns.columns:
        col_data = returns[col].dropna()
        if len(col_data) > 0:
            # Check for extreme returns (> 100% daily return)
            extreme_returns = (col_data.abs() > 1.0).sum()
            if extreme_returns > 0:
                validation_results['warnings'].append(f"{col}: {extreme_returns} extreme returns (>100% daily)")
    
    # Check data frequency
    date_diffs = returns.index.to_series().diff().dropna()
    most_common_freq = date_diffs.mode()
    if len(most_common_freq) > 0:
        validation_results['stats']['most_common_frequency'] = str(most_common_freq[0])
    
    # Check for sufficient data
    if len(returns) < 100:
        validation_results['warnings'].append(f"Limited data: only {len(returns)} observations")
    
    return validation_results


def preprocess_returns_data(returns, fill_method='forward', remove_outliers=True, outlier_threshold=5.0):
    """
    Preprocess returns data by handling missing values and outliers.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Raw returns data
    fill_method : str
        Method for filling missing values ('forward', 'backward', 'zero', 'drop')
    remove_outliers : bool
        Whether to cap extreme outliers
    outlier_threshold : float
        Number of standard deviations for outlier detection
        
    Returns
    -------
    pd.DataFrame
        Preprocessed returns data
    """
    print("Preprocessing returns data...")
    
    processed_returns = returns.copy()
    
    # Handle missing values
    if fill_method == 'forward':
        processed_returns = processed_returns.fillna(method='ffill')
    elif fill_method == 'backward':
        processed_returns = processed_returns.fillna(method='bfill')
    elif fill_method == 'zero':
        processed_returns = processed_returns.fillna(0)
    elif fill_method == 'drop':
        processed_returns = processed_returns.dropna()
    
    # Remove outliers if requested
    if remove_outliers:
        for col in processed_returns.columns:
            col_data = processed_returns[col]
            mean_val = col_data.mean()
            std_val = col_data.std()
            
            if std_val > 0:
                # Cap values beyond threshold standard deviations
                lower_bound = mean_val - outlier_threshold * std_val
                upper_bound = mean_val + outlier_threshold * std_val
                
                outliers_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                if outliers_count > 0:
                    print(f"Capping {outliers_count} outliers in {col}")
                    processed_returns[col] = col_data.clip(lower_bound, upper_bound)
    
    print(f"Preprocessing complete: {len(processed_returns)} rows, {len(processed_returns.columns)} assets")
    
    return processed_returns
