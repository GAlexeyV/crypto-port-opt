"""
Momentum-HRP Hybrid Strategy

This strategy combines two approaches:
1. Momentum: Select top 20 assets based on recent performance (momentum)
2. HRP (Hierarchical Risk Parity): Apply risk-based weighting to these selected assets

The goal is to capture both momentum effects and optimize for risk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

# Import necessary functions from utilities and momentum_strategy.py
from ..utils.hrp_utils import get_hrp_weights

def run_momentum_hrp_backtest(returns_data, lookback_days=60, momentum_window=30, 
                             rebalancing_freq='W', initial_capital=100000, n_momentum_assets=20, min_allocation=0.02):
    """
    Run a backtest on a momentum-HRP hybrid strategy.
    
    Parameters:
    - returns_data: DataFrame with asset returns
    - lookback_days: Number of days to look back for calculating covariance and correlation
    - momentum_window: Number of days to look back for calculating momentum
    - rebalancing_freq: Frequency of rebalancing ('W' for weekly, 'M' for monthly)
    - initial_capital: Initial capital to invest
    - n_momentum_assets: Number of top assets to select based on momentum
    - min_allocation: Minimum allocation to each asset
    
    Returns:
    - portfolio_values: Series of portfolio values over time
    - weights_history: Dictionary of weights at each rebalancing date
    """
    available_dates = returns_data.index.sort_values()
    
    # Ensure we have enough lookback data
    if lookback_days >= len(available_dates):
        lookback_days = int(len(available_dates) / 2)  # Use half of available data if not enough
        print(f"Warning: Not enough data for specified lookback. Using {lookback_days} days instead.")
    
    # Select dates after the lookback period
    start_date = available_dates[lookback_days]
    test_dates = available_dates[available_dates >= start_date]
    
    # Determine rebalancing dates
    if rebalancing_freq == 'W':
        rebalance_dates = pd.date_range(start=start_date, end=available_dates[-1], freq='W-FRI')
    elif rebalancing_freq == 'M':
        rebalance_dates = pd.date_range(start=start_date, end=available_dates[-1], freq='M')
    else:
        raise ValueError("rebalancing_freq must be 'W' or 'M'")
    
    # Filter to ensure rebalance dates exist in available dates
    rebalance_dates = [date for date in rebalance_dates if date in available_dates]
    
    # Initialize tracking variables
    portfolio_values = pd.Series(index=test_dates, dtype=float)
    portfolio_values.iloc[0] = initial_capital
    portfolio_value = initial_capital
    weights_history = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
    current_weights = pd.Series(0, index=returns_data.columns)
    last_rebalance_date = None
    
    print(f"\nInitialized portfolio tracking: ")
    print(f"  - Start date: {test_dates[0]}")
    print(f"  - End date: {test_dates[-1]}")
    print(f"  - Portfolio values index has {len(portfolio_values)} dates")
    
    # Main backtest loop - iterate through dates
    for i in range(1, len(returns_data)):
        current_date = returns_data.index[i]
        
        # Check if we should rebalance (weekly - Sunday)
        if last_rebalance_date is None or \
           (current_date.weekday() == 6 and (current_date - last_rebalance_date).days >= 5):
            
            # Get lookback window up to current date (no look-ahead bias)
            start_date = current_date - pd.Timedelta(days=lookback_days + 30)
            lookback_window = returns_data.loc[start_date:current_date]
            
            if len(lookback_window) < lookback_days:
                continue
                
            # Dynamic asset selection based on data availability
            min_required_data = int(lookback_days * 0.8)
            available_assets = []
            
            for asset in lookback_window.columns:
                asset_data = lookback_window[asset].dropna()
                if len(asset_data) >= min_required_data:
                    # Check if asset has recent data (within last 7 days)
                    recent_data = lookback_window[asset].tail(7).dropna()
                    if len(recent_data) > 0:
                        available_assets.append(asset)
            
            if len(available_assets) < 3:  # Need at least 3 assets
                continue
                
            print(f"Date: {current_date.date()}, Available assets: {len(available_assets)}")
            
            # Use only available assets for this rebalancing period
            current_returns = lookback_window[available_assets].dropna()
            
            if len(current_returns) < max(30, lookback_days * 0.6):
                continue
            
            # Calculate momentum scores for available assets (cumulative returns)
            momentum_scores = (current_returns.rolling(momentum_window).apply(lambda x: (1 + x).prod() - 1) * 252/momentum_window)
            latest_momentum = momentum_scores.iloc[-1].dropna()
            
            if len(latest_momentum) < 3:
                continue
                
            # Select top momentum assets from available universe
            top_momentum_assets = latest_momentum.nlargest(n_momentum_assets).index.tolist()
            top_momentum_assets = [asset for asset in top_momentum_assets if asset in available_assets]
            
            if len(top_momentum_assets) < 3:
                continue
                
            print(f"Top momentum assets: {', '.join(top_momentum_assets[:3])}...")
            
            # Calculate covariance and correlation for momentum-selected assets
            momentum_returns = current_returns[top_momentum_assets]
            cov = momentum_returns.cov() * 252
            corr = momentum_returns.corr()
            
            # Validate matrices
            if cov.isnull().any().any() or corr.isnull().any().any():
                print(f"Warning: Invalid matrices on {current_date.date()}")
                continue
            
            try:
                # Calculate HRP weights for momentum-selected assets
                weights = get_hrp_weights(
                    cov, 
                    corr, 
                    returns_data=momentum_returns,
                    n_clusters=min(6, len(top_momentum_assets) // 2),
                    n_per_cluster=2,
                    min_allocation=min_allocation
                )
                
                if weights.sum() > 0:
                    # Update current weights
                    current_weights = pd.Series(0.0, index=returns_data.columns)
                    for asset in weights.index:
                        if asset in current_weights.index:
                            current_weights[asset] = weights[asset]
                    
                    last_rebalance_date = current_date
                    active_positions = (current_weights > 0.001).sum()
                    print(f"Rebalanced to {active_positions} positions")
                    
            except Exception as e:
                print(f"HRP calculation failed on {current_date.date()}: {str(e)}")
                continue
            
            # Record weights for this date
            weights_history.loc[current_date] = current_weights
        
        # Update portfolio value based on daily returns
        if i > 0:  # Skip the first day since we don't have previous weights
            daily_returns = returns_data.loc[current_date]
            # Calculate portfolio return as weighted sum of asset returns (only for assets with both weights and returns)
            valid_assets = daily_returns.dropna().index.intersection(current_weights[current_weights > 0].index)
            if len(valid_assets) > 0:
                portfolio_return = (current_weights[valid_assets] * daily_returns[valid_assets]).sum()
                # Update portfolio value
                portfolio_value *= (1 + portfolio_return)
            
            # Note: We DO NOT update weights daily - only on rebalancing dates
            # This prevents artificial daily rebalancing that inflates returns
        
        # Store portfolio value
        portfolio_values[current_date] = portfolio_value
        
        # Debug output for last 5 days
        if i > len(returns_data) - 5:
            print(f"\n[DEBUG] Date: {current_date}, Portfolio value: {portfolio_value:,.2f}")
            print(f"         Index position in returns_data: {i} of {len(returns_data)}")
    
    return portfolio_values, weights_history

def calculate_performance_metrics(portfolio_values):
    """Calculate key performance metrics."""
    # Sort portfolio values by date to ensure proper time order
    portfolio_values = portfolio_values.sort_index()
    
    # Initial and final portfolio values
    initial_value = portfolio_values.iloc[0]
    final_value = portfolio_values.iloc[-1]
    
    print(f"\n[DEBUG] Performance metrics calculation:")
    print(f"  - Portfolio values shape: {portfolio_values.shape}")
    print(f"  - First date: {portfolio_values.index[0]}, value: {portfolio_values.iloc[0]:,.2f}")
    print(f"  - Last date: {portfolio_values.index[-1]}, value: {portfolio_values.iloc[-1]:,.2f}")
    
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
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio
    }

def print_performance_summary(portfolio_values, weights):
    """Print performance metrics summary."""
    metrics = calculate_performance_metrics(portfolio_values)
    
    print("\n=== Momentum-HRP Hybrid Strategy Performance ===")
    print(f"Initial Capital: ${portfolio_values.iloc[0]:,.2f}")
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Volatility (Annualized): {metrics['volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Print final allocation
    non_zero_weights = weights.loc[weights.sum(axis=1) > 0]
    if len(non_zero_weights) > 0:
        last_date = non_zero_weights.index[-1]
        final_weights = weights.loc[last_date]
        active_positions = sum(final_weights > 0)
        
        print(f"\nFinal portfolio allocation (date: {last_date})")
        print(f"Number of active positions: {active_positions}")
        
        # Print top holdings
        top_holdings = final_weights[final_weights > 0].sort_values(ascending=False)
        print("\nTop holdings:")
        for asset, weight in top_holdings.head(10).items():
            print(f"{asset}: {weight*100:.2f}%")

def plot_portfolio_value(portfolio_values, save_path=None):
    """Plot portfolio value over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values.index, portfolio_values, label='Portfolio Value')
    plt.title('Momentum-HRP Hybrid Strategy Portfolio Value', fontsize=14)
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

if __name__ == "__main__":
    # Set warning filter
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Load returns data
    returns_file = "../../data/daily_returns.csv"
    print(f"Loading returns data from {returns_file}")
    returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    
    # Set parameters
    initial_capital = 100_000
    lookback_days = 60  # General lookback window
    momentum_window = 30  # Specific window for momentum calculation
    n_momentum_assets = 20  # Select top 20 assets by momentum
    min_allocation = 0.02  # Set 2% minimum allocation
    
    # Run the momentum-HRP hybrid backtest
    print("\n=== Running Momentum-HRP Hybrid Strategy Backtest (Weekly Rebalancing) ===")
    hybrid_values, hybrid_weights = run_momentum_hrp_backtest(
        returns, 
        lookback_days=lookback_days,
        momentum_window=momentum_window,
        initial_capital=initial_capital, 
        rebalancing_freq='W',
        n_momentum_assets=n_momentum_assets,
        min_allocation=min_allocation
    )
    
    # Print performance results
    print_performance_summary(hybrid_values, hybrid_weights)
    
    # Plot portfolio value
    plot_portfolio_value(hybrid_values, save_path='../../images/momentum_hrp_portfolio.png')
    
    # Debug portfolio values
    print(f"\n=== DEBUG: Portfolio Values Analysis ===")
    print(f"Portfolio values shape: {hybrid_values.shape}")
    print(f"First few values:")
    print(hybrid_values.head())
    print(f"Last few values:")
    print(hybrid_values.tail())
    print(f"Final value from iloc[-1]: {hybrid_values.iloc[-1]}")
    print(f"Date range: {hybrid_values.index[0]} to {hybrid_values.index[-1]}")
    
    # Debug portfolio values before saving
    print(f"\n=== DEBUG: Portfolio Values Before Saving ===")
    print(f"Portfolio values shape: {hybrid_values.shape}")
    print(f"First few dates:")
    for i in range(5):
        if i < len(hybrid_values):
            print(f"  {hybrid_values.index[i]}: {hybrid_values.iloc[i]:,.2f}")
            
    print(f"Last few dates:")
    for i in range(1, 6):
        if i <= len(hybrid_values):
            print(f"  {hybrid_values.index[-i]}: {hybrid_values.iloc[-i]:,.2f}")
    
    # Check for any sorting issues
    sorted_values = hybrid_values.sort_index()
    if not sorted_values.equals(hybrid_values):
        print("WARNING: Portfolio values are not sorted by date!")
        hybrid_values = sorted_values
        print("Values have been sorted")
        
    # Sort portfolio values by date before saving
    hybrid_values = hybrid_values.sort_index()
    
    # Save portfolio values for comparison
    hybrid_values.to_csv('../../data/momentum_hrp_portfolio_values.csv')
    print("\nSaved portfolio values to 'data/momentum_hrp_portfolio_values.csv'")
