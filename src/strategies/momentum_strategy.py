"""
Momentum trading strategy for crypto portfolio backtest

This file implements a momentum-based strategy that:
1. Calculates momentum scores for all assets
2. Selects the top N assets with the highest momentum
3. Allocates funds equally among the selected assets
4. Rebalances on a specified frequency (default: weekly)

The momentum score is based on the asset's return over a specified lookback period.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

def calculate_momentum(returns_data, lookback_days=30):
    """
    Calculate momentum scores for each asset.
    
    Parameters:
    -----------
    returns_data: pandas.DataFrame
        DataFrame of daily returns for each asset
    lookback_days: int
        Number of days to look back for momentum calculation
    
    Returns:
    --------
    pandas.Series
        Series of momentum scores for each asset
    """
    # Calculate cumulative returns over the lookback period
    cumulative_returns = (1 + returns_data.iloc[-lookback_days:]).prod() - 1
    
    return cumulative_returns

def get_momentum_weights(returns_data, lookback_days=30, n_assets=10):
    """
    Calculate portfolio weights using momentum strategy.
    
    Parameters:
    -----------
    returns_data: pandas.DataFrame
        DataFrame of daily returns for each asset
    lookback_days: int
        Number of days to look back for momentum calculation
    n_assets: int
        Number of top assets to select
    
    Returns:
    --------
    pandas.Series
        Series of portfolio weights
    """
    if len(returns_data) < lookback_days:
        # Not enough data, return equal weights for all assets
        return pd.Series(1/len(returns_data.columns), index=returns_data.columns)
    
    # Calculate momentum for each asset
    momentum_scores = calculate_momentum(returns_data, lookback_days)
    
    # Rank assets by momentum and select top N
    top_assets = momentum_scores.nlargest(n_assets).index
    
    # Create weights (equal weight among top assets)
    weights = pd.Series(0, index=returns_data.columns)
    weights[top_assets] = 1 / len(top_assets)
    
    return weights

def run_momentum_backtest(returns_data, lookback_days=30, momentum_window=30, 
                         rebalancing_freq='W', initial_capital=100000, n_assets=10):
    """
    Run a backtest for a momentum portfolio strategy.
    
    Parameters:
    -----------
    returns_data: pandas.DataFrame
        DataFrame of daily returns for each asset
    lookback_days: int
        Number of days of history to use for calculations
    momentum_window: int
        Number of days to look back for momentum calculation
    rebalancing_freq: str
        Pandas frequency string for rebalancing schedule ('W' for weekly, 'M' for monthly)
    initial_capital: float
        Initial portfolio value
    n_assets: int
        Number of top momentum assets to include
    
    Returns:
    --------
    tuple
        (portfolio_values, weights_history)
    """
    # Initialize the portfolio
    portfolio_value = initial_capital
    portfolio_values = pd.Series(index=returns_data.index)
    portfolio_values.iloc[0] = portfolio_value
    
    # Create a dictionary to store weights history
    weights_history = {}
    
    # Get rebalance dates (using the last day of each week/month)
    # Make sure we're only using dates that exist in the dataset
    rebalance_dates = returns_data.resample(rebalancing_freq).last().index
    rebalance_dates = [date for date in rebalance_dates if date in returns_data.index]
    
    # Current portfolio state
    current_weights = pd.Series(0, index=returns_data.columns)
    
    for i, date in enumerate(rebalance_dates):
        # Check if we have enough data history for this rebalance date
        date_loc = returns_data.index.get_loc(date)
        if date_loc < lookback_days:
            print(f"Skipping rebalance on {date}: Not enough historical data.")
            continue
            
        # Get lookback window for this rebalance date
        lookback_window = returns_data.iloc[date_loc-lookback_days+1:date_loc+1]
        
        # Calculate momentum-based weights
        current_weights = get_momentum_weights(lookback_window, momentum_window, n_assets)
        
        # Store the weights
        weights_history[date] = current_weights
        
        # Display rebalance information
        active_positions = sum(current_weights > 0)
        top_holdings = current_weights[current_weights > 0].sort_values(ascending=False).head(3)
        
        print(f"Date: {date}, Active positions: {active_positions}")
        
        if len(top_holdings) > 0:
            top_names = ', '.join([f"{idx}" for idx in top_holdings.index])
            print(f"Top holdings: {top_names}")
            
        # Update the portfolio value until the next rebalance date or the end of the data
        next_idx = i + 1
        if next_idx < len(rebalance_dates):
            next_date = rebalance_dates[next_idx]
            date_range = returns_data.loc[date:next_date].index
        else:
            date_range = returns_data.loc[date:].index
            
        # Calculate portfolio returns for each day in this period
        for j in range(1, len(date_range)):
            current_date = date_range[j-1]
            next_date = date_range[j]
            
            # Get the return for this day
            day_return = returns_data.loc[next_date]
            
            # Calculate portfolio return (weighted sum of asset returns)
            portfolio_return = (current_weights * day_return).sum()
            
            # Update portfolio value
            portfolio_value = portfolio_value * (1 + portfolio_return)
            portfolio_values.loc[next_date] = portfolio_value
            
            # No rebalancing within the period, so weights change with asset price movements
            # Update weights based on asset returns (this mimics real portfolio behavior)
            new_values = current_weights * (1 + day_return)
            # Avoid division by zero if all assets have 0 value
            if new_values.sum() > 0:
                current_weights = new_values / new_values.sum()  # Normalize to 100%
    
    # Forward fill any NA values in portfolio values (for days with no trading)
    portfolio_values = portfolio_values.ffill()
    
    # Convert weights_history dictionary to DataFrame for compatibility with print_performance_summary
    weights_df = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
    for date, weights in weights_history.items():
        weights_df.loc[date] = weights
    weights_df = weights_df.fillna(0)  # Fill NaN with zeros
    
    return portfolio_values, weights_df

def calculate_performance_metrics(portfolio_values):
    """
    Calculate key performance metrics for a portfolio.
    
    Parameters:
    -----------
    portfolio_values: pandas.Series
        Series of portfolio values over time
    
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    initial_value = portfolio_values.iloc[0]
    final_value = portfolio_values.iloc[-1]
    
    # Calculate returns
    returns = portfolio_values.pct_change().dropna()
    
    # Total return
    total_return = (final_value / initial_value) - 1
    
    # Annualized return
    days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Maximum drawdown
    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values / rolling_max) - 1
    max_drawdown = drawdown.min()
    
    # Calculate additional risk metrics
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    
    metrics = {
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio
    }
    
    return metrics

def print_performance_summary(portfolio_values, weights_history):
    """
    Print a summary of portfolio performance and final allocation.
    
    Parameters:
    -----------
    portfolio_values: pandas.Series
        Series of portfolio values over time
    weights_history: pandas.DataFrame
        DataFrame of portfolio weights over time
    """
    metrics = calculate_performance_metrics(portfolio_values)
    
    print(f"Initial Capital: ${portfolio_values.iloc[0]:,.2f}")
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Volatility (Annualized): {metrics['volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Print final portfolio allocation
    # Find the last date with non-zero weights (the last rebalance date)
    non_zero_dates = weights_history.sum(axis=1)
    non_zero_dates = non_zero_dates[non_zero_dates > 0]
    
    if len(non_zero_dates) > 0:
        final_date = non_zero_dates.index[-1]
        final_weights = weights_history.loc[final_date]
        active_positions = final_weights[final_weights > 0]
    else:
        final_date = weights_history.index[-1]
        final_weights = pd.Series(0, index=weights_history.columns)
        active_positions = pd.Series()
    
    print(f"\nFinal portfolio allocation (date: {final_date})")
    print(f"Number of active positions: {len(active_positions)}")
    
    print("\nTop holdings:")
    for symbol, weight in active_positions.sort_values(ascending=False).head(10).items():
        print(f"{symbol}: {weight*100:.2f}%")

if __name__ == "__main__":
    # Load returns data
    returns_file = "../../data/daily_returns.csv"
    
    print(f"Loading returns data from {returns_file}")
    returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    
    # Set parameters
    initial_capital = 100_000
    lookback_days = 60  # General lookback window
    momentum_window = 30  # Specific window for momentum calculation
    n_assets = 10  # Select top 10 assets
    
    # Run the momentum backtest
    print("\n=== Running Momentum Strategy Backtest (Weekly Rebalancing) ===")
    momentum_values, momentum_weights = run_momentum_backtest(
        returns, 
        lookback_days=lookback_days,
        momentum_window=momentum_window,
        initial_capital=initial_capital, 
        rebalancing_freq='W',
        n_assets=n_assets
    )
    
    # Print momentum performance results
    print("\n=== Momentum Strategy Performance ===")
    print_performance_summary(momentum_values, momentum_weights)
    
    # Save momentum results to file for later comparison
    momentum_values.to_csv('../../data/momentum_portfolio_values.csv')
    
    # Visualize momentum performance
    plt.figure(figsize=(12, 6))
    plt.plot(momentum_values.index, momentum_values, label='Momentum Portfolio')
    plt.title('Momentum Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../images/momentum_performance.png')
    plt.close()
