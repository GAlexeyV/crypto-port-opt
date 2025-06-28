#!/usr/bin/env python3
# Equal Weight Portfolio Strategy for comparison with HRP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def get_equal_weights(returns_data, n_assets=None):
    """
    Creates equal-weighted portfolio with all available assets or top N assets.
    
    Parameters:
    -----------
    returns_data: pandas.DataFrame
        DataFrame of asset returns
    n_assets: int, optional
        Number of assets to include (if None, includes all assets)
    
    Returns:
    --------
    pandas.Series
        Equal weights for all assets
    """
    assets = returns_data.columns.tolist()
    
    # If n_assets is provided, use only that many assets
    if n_assets is not None and n_assets < len(assets):
        # Calculate Sharpe ratio to pick top performing assets
        mean_returns = returns_data.mean() * 252  # Annualized returns
        std_returns = returns_data.std() * np.sqrt(252)  # Annualized volatility
        sharpe = mean_returns / std_returns
        
        # Select top n_assets by Sharpe ratio
        assets = sharpe.sort_values(ascending=False).index[:n_assets].tolist()
    
    # Create equal weights
    weight = 1.0 / len(assets)
    weights = pd.Series(weight, index=assets)
    
    # Add zero weights for any assets not in the selected set
    all_assets = returns_data.columns
    full_weights = pd.Series(0, index=all_assets)
    full_weights.loc[weights.index] = weights
    
    return full_weights.sort_index()

def run_equal_weight_backtest(returns_data, lookback_days=60, rebalancing_freq='W', 
                             initial_capital=100000, n_assets=None):
    """
    Run a backtest for an equal-weighted portfolio strategy.
    
    Parameters:
    -----------
    returns_data: pandas.DataFrame
        DataFrame of daily returns for each asset
    lookback_days: int
        Number of days of history to use for calculations
    rebalancing_freq: str
        Pandas frequency string for rebalancing schedule ('W' for weekly, 'M' for monthly)
    initial_capital: float
        Initial portfolio value
    n_assets: int, optional
        Number of assets to include (if None, includes all assets)
    
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
        
        # Calculate equal weights
        current_weights = get_equal_weights(lookback_window, n_assets)
        
        # Store the weights
        weights_history[date] = current_weights
        
        # Display rebalance information
        active_positions = sum(current_weights > 0)
        print(f"Date: {date}, Active positions: {active_positions}")
            
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
    
    # Run the equal weight backtest
    initial_capital = 100_000
    lookback_days = 60  # Same as HRP
    n_assets = None  # Use all available assets
    
    print("\n=== Running Equal Weight Backtest (Weekly Rebalancing) ===")
    portfolio_values, weights_history = run_equal_weight_backtest(
        returns, 
        lookback_days=lookback_days,
        initial_capital=initial_capital, 
        rebalancing_freq='W',
        n_assets=n_assets
    )
    
    # Print performance results
    print("\n=== Equal Weight Portfolio Performance ===")
    print_performance_summary(portfolio_values, weights_history)
    
    # Visualize portfolio performance
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values.index, portfolio_values, label='Equal Weight Portfolio')
    
    # Try to run the main HRP backtest for comparison
    try:
        # Note: HRP comparison functionality moved to strategy_comparison.py
        # This legacy comparison code is commented out
        pass
    except:
        pass  # Skip if not available
    
    # Format the plot
    plt.title('Equal Weight Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../../images/equal_weight_performance.png')
    plt.show()
    
    # Save portfolio values to CSV for strategy comparison
    csv_file = '../../data/equal_weight_portfolio_values.csv'
    portfolio_values.to_csv(csv_file)
    print(f"\nSaved portfolio values to 'data/equal_weight_portfolio_values.csv'")
