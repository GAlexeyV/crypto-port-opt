"""
Hierarchical Risk Parity (HRP) Strategy

This strategy applies the Hierarchical Risk Parity algorithm developed by Marcos Lopez de Prado.
HRP builds portfolios based on hierarchical clustering of the correlation matrix between assets,
providing better diversification than simple risk parity or minimum variance approaches.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import HRP utility functions
from ..utils.hrp_utils import get_hrp_weights, filter_valid_symbols_for_window, validate_returns_for_hrp, clean_correlation_matrix
from ..utils.covariance_estimators import sample_covariance

def run_hrp_backtest(returns_data, lookback_days=60, rebalancing_freq='W', initial_capital=100000):
    """
    Run a backtest on a Hierarchical Risk Parity (HRP) portfolio allocation strategy.
    Uses dynamic asset selection based on data availability at each rebalancing date.
    
    Parameters
    ----------
    returns_data : pandas.DataFrame
        DataFrame containing daily returns for each asset
    lookback_days : int
        Number of days to look back for calculating correlation and covariance
    rebalancing_freq : str
        Rebalancing frequency, 'W' for weekly, 'M' for monthly
    initial_capital : float
        Initial capital to invest
        
    Returns
    -------
    portfolio_values : pandas.Series
        Series of portfolio values over time
    weights_history : pandas.DataFrame
        DataFrame containing portfolio weights over time
    """
    print(f"Running HRP strategy backtest with {lookback_days}-day lookback and {rebalancing_freq} rebalancing")
    
    # Convert index to datetime if it's not already
    if not isinstance(returns_data.index, pd.DatetimeIndex):
        returns_data.index = pd.to_datetime(returns_data.index)
        
    # Sort the returns data by date
    returns_data = returns_data.sort_index()
    
    # Initialize portfolio tracking variables
    portfolio_values = pd.Series(index=returns_data.index, dtype=float)
    portfolio_values.iloc[0] = initial_capital
    # Initialize weights_history with float dtype to prevent dtype warnings
    weights_history = pd.DataFrame(0.0, index=returns_data.index, columns=returns_data.columns, dtype=float)
    
    # Set the initial date and portfolio value
    current_date = returns_data.index[0]
    current_value = initial_capital
    
    # Current portfolio weights (start with no positions)
    current_weights = pd.Series(0, index=returns_data.columns)
    last_rebalance_date = None
    
    # Generate rebalancing dates - Fixed for proper weekly rebalancing  
    if rebalancing_freq == 'W':
        # Start from first Sunday after the start date
        start_date = returns_data.index[0]
        first_sunday = start_date + pd.Timedelta(days=(6 - start_date.weekday()) % 7)
        rebalancing_dates = pd.date_range(start=first_sunday, end=returns_data.index[-1], freq='W-SUN')
    else:  # Monthly or other frequencies
        rebalancing_dates = pd.date_range(start=returns_data.index[0], end=returns_data.index[-1], freq=rebalancing_freq)
    
    # Filter to ensure rebalance dates exist in available dates
    rebalancing_dates = [date for date in rebalancing_dates if date in returns_data.index]
    
    log_messages = []
    
    # Iterate through all dates for daily portfolio updates
    for i, current_date in enumerate(returns_data.index):
        
        # Check if this is a rebalancing date
        if current_date in rebalancing_dates:
            # Get lookback window up to current rebalance date (no look-ahead bias)
            start_date = current_date - pd.Timedelta(days=lookback_days + 30)
            lookback_window = returns_data.loc[start_date:current_date]
            
            if len(lookback_window) < lookback_days:
                # Skip rebalancing but don't record weights - just continue with current weights
                if i > 0:
                    daily_return = (returns_data.loc[current_date] * current_weights).sum()
                    current_value = current_value * (1 + daily_return)
                portfolio_values.loc[current_date] = current_value
                continue
                
            # Select only assets with sufficient data availability at this point in time
            min_required_data = int(lookback_days * 0.8)
            available_assets = []
            
            for asset in lookback_window.columns:
                asset_data = lookback_window[asset].dropna()
                if len(asset_data) >= min_required_data:
                    # Check if asset has recent data (within last 7 days)
                    recent_data = lookback_window[asset].tail(7).dropna()
                    if len(recent_data) > 0:
                        available_assets.append(asset)
            
            if len(available_assets) < 3:  # Need at least 3 assets for HRP
                # No valid HRP possible - keep current weights, don't record new weights
                log_messages.append(f"Date: {current_date.date()}, No rebalancing - insufficient assets ({len(available_assets)})")
                if i > 0:
                    daily_return = (returns_data.loc[current_date] * current_weights).sum()
                    current_value = current_value * (1 + daily_return)
                portfolio_values.loc[current_date] = current_value
                continue
                
            # We have sufficient assets - now calculate new HRP weights
            log_messages.append(f"Rebalancing on {current_date.date()} with {len(available_assets)} available assets")
            
            # Use only available assets for this rebalancing period
            current_returns = lookback_window[available_assets].dropna()
            
            if len(current_returns) < max(30, lookback_days * 0.6):
                # Insufficient data for reliable calculation - keep current weights
                log_messages.append(f"Date: {current_date.date()}, No rebalancing - insufficient data points ({len(current_returns)})")
                if i > 0:
                    daily_return = (returns_data.loc[current_date] * current_weights).sum()
                    current_value = current_value * (1 + daily_return)
                portfolio_values.loc[current_date] = current_value
                continue
                
            # Clean and validate the returns data
            try:
                current_returns = validate_returns_for_hrp(current_returns)
                
                # Calculate annualized covariance and correlation matrices
                cov = current_returns.cov() * 252
                corr = current_returns.corr()
                
                # Get dynamic parameters
                n_clusters = min(10, len(current_returns.columns) // 3)
                n_per_cluster = 2
                
                # Calculate HRP weights
                weights = get_hrp_weights(
                    cov=cov, 
                    corr=corr,
                    returns_data=current_returns,
                    n_clusters=n_clusters,
                    n_per_cluster=n_per_cluster,
                    min_allocation=0.02
                )
                
                if weights is None or weights.sum() == 0:
                    log_messages.append(f"Date: {current_date.date()}, HRP weights calculation returned empty - keeping current weights")
                    if i > 0:
                        daily_return = (returns_data.loc[current_date] * current_weights).sum()
                        current_value = current_value * (1 + daily_return)
                    portfolio_values.loc[current_date] = current_value
                    continue
                    
                # Update current weights (only for available assets)
                new_weights = pd.Series(0.0, index=returns_data.columns)
                for asset in weights.index:
                    if asset in new_weights.index:
                        new_weights[asset] = weights[asset]
                        
                current_weights = new_weights
                active_positions = (current_weights > 0.001).sum()
                log_messages.append(f"Successfully rebalanced to {active_positions} positions on {current_date.date()}")
                
                # ONLY record weights when we actually rebalance successfully
                weights_history.loc[current_date] = current_weights
                
            except Exception as e:
                log_messages.append(f"HRP calculation failed on {current_date.date()}: {str(e)}")
                # Keep previous weights - no weight recording
                pass
                
        # Update portfolio value based on daily returns (skip first day)
        if i > 0:
            daily_return = (returns_data.loc[current_date] * current_weights).sum()
            current_value = current_value * (1 + daily_return)
        
        portfolio_values.loc[current_date] = current_value
        
    return portfolio_values, weights_history

def plot_portfolio_value(portfolio_values, save_path=None):
    """Plot portfolio value over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values.index, portfolio_values, label='Portfolio Value')
    plt.title('HRP Strategy Portfolio Value', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def print_performance_summary(portfolio_values, weights_history=None):
    """Print a summary of the strategy's performance."""
    from ..analysis.compare_strategies import calculate_metrics
    
    # Calculate performance metrics
    metrics = calculate_metrics(portfolio_values)
    
    # Print the summary
    print("\n=== HRP Strategy Performance ===")
    print(f"Initial Portfolio Value: ${metrics['initial_value']:,.2f}")
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Volatility (Annualized): {metrics['volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    
    if weights_history is not None:
        # Find the last date with non-zero weights
        non_zero_dates = weights_history.sum(axis=1)
        non_zero_dates = non_zero_dates[non_zero_dates > 0]
        
        if len(non_zero_dates) > 0:
            final_date = non_zero_dates.index[-1]
            final_weights = weights_history.loc[final_date]
            active_positions = final_weights[final_weights > 0]
            
            print(f"\nFinal portfolio allocation (date: {final_date})")
            print(f"Number of active positions: {len(active_positions)}")
            
            print("\nTop holdings:")
            for symbol, weight in active_positions.sort_values(ascending=False).head(10).items():
                print(f"{symbol}: {weight*100:.2f}%")

if __name__ == "__main__":
    import warnings
    # Set warning filter
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Load returns data
    returns_file = "../../data/daily_returns.csv"
    print(f"Loading returns data from {returns_file}")
    returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    
    # Run the HRP backtest
    print("\n=== Running HRP Strategy Backtest (Weekly Rebalancing) ===")
    portfolio_values, weights_history = run_hrp_backtest(
        returns, 
        lookback_days=60,
        initial_capital=100000, 
        rebalancing_freq='W'
    )
    
    # Print performance results
    print_performance_summary(portfolio_values, weights_history)
    
    # Plot portfolio value
    plot_portfolio_value(portfolio_values, save_path='../../images/hrp_portfolio.png')
    
    # Save portfolio values for comparison
    portfolio_values.to_csv('../../data/hrp_portfolio_values.csv')
    print("\nSaved portfolio values to 'data/hrp_portfolio_values.csv'")
