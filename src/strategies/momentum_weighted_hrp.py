"""
Momentum-Weighted HRP Strategy

This strategy enhances the hybrid approach by:
1. Momentum: Select top 20 assets based on recent performance (momentum)
2. HRP (Hierarchical Risk Parity): Apply risk-based weighting to these selected assets
3. Key Enhancement: Weight the final allocations by the momentum scores

The goal is to maintain risk-parity benefits while tilting more capital toward stronger momentum assets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

# Import necessary functions from utilities and momentum_strategy.py
from ..utils.hrp_utils import get_hrp_weights

def run_momentum_weighted_hrp_backtest(returns_data, lookback_days=60, momentum_window=30, 
                             rebalancing_freq='W', initial_capital=100000, n_momentum_assets=20,
                             momentum_weight=0.5):
    """
    Run a backtest on a momentum-weighted HRP hybrid strategy.
    
    Parameters:
    - returns_data: DataFrame with asset returns
    - lookback_days: Number of days to look back for calculating covariance and correlation
    - momentum_window: Number of days to look back for calculating momentum
    - rebalancing_freq: Frequency of rebalancing ('W' for weekly, 'M' for monthly)
    - initial_capital: Initial capital to invest
    - n_momentum_assets: Number of top assets to select based on momentum
    - momentum_weight: Weight given to momentum scores (0-1) when blending with HRP weights
    
    Returns:
    - portfolio_values: Series of portfolio values over time
    - weights_history: DataFrame of weights at each rebalancing date
    """
    available_dates = returns_data.index.sort_values()
    
    # Ensure we have enough lookback data
    if lookback_days >= len(available_dates):
        lookback_days = int(len(available_dates) / 2)  # Use half of available data if not enough
        print(f"Warning: Not enough data for specified lookback. Using {lookback_days} days instead.")
    
    # Find the first date where we have enough data
    start_idx = max(lookback_days, momentum_window)
    test_dates = available_dates[start_idx:]
    
    # Set rebalancing dates
    if rebalancing_freq == 'W':
        # Weekly rebalancing on Sundays (weekday 6)
        rebalance_dates = pd.date_range(start=test_dates[0], end=available_dates[-1], freq='W-SUN')
    elif rebalancing_freq == 'M':
        rebalance_dates = pd.date_range(start=test_dates[0], end=available_dates[-1], freq='M')
    else:
        raise ValueError("rebalancing_freq must be 'W' or 'M'")
    
    # Filter to ensure rebalance dates exist in available dates
    rebalance_dates = [date for date in rebalance_dates if date in available_dates]
    
    # Initialize tracking variables
    portfolio_values = pd.Series(index=returns_data.index, dtype=float)
    portfolio_values.iloc[0] = initial_capital
    portfolio_value = initial_capital
    weights_history = {}
    current_weights = pd.Series(0.0, index=returns_data.columns, dtype=float)
    
    # Main backtest loop - iterate through all dates
    for i, current_date in enumerate(returns_data.index):
        
        # Rebalance on Sundays or first day
        if current_date.weekday() == 6 or i == 0:  # Sunday rebalancing or first day
            
            # Skip first few days if insufficient data for momentum calculation
            if i < momentum_window:
                portfolio_values[current_date] = portfolio_value
                continue
                
            # Get lookback window up to current date (no look-ahead bias)
            start_date = current_date - pd.Timedelta(days=lookback_days + 30)
            lookback_window = returns_data.loc[start_date:current_date]
            
            if len(lookback_window) < lookback_days:
                continue
                
            # Dynamic asset selection based on data availability
            # For momentum: need momentum_window + buffer for rolling calculation
            # For HRP: need sufficient recent data for correlation calculation
            min_required_data = int(lookback_days * 0.8)
            momentum_min_data = momentum_window + 5  # Need extra buffer for rolling calculation
            available_assets = []
            
            for asset in lookback_window.columns:
                asset_data = lookback_window[asset].dropna()
                
                # Check if asset has enough data for HRP (correlation calculation)
                if len(asset_data) >= min_required_data:
                    # Check if asset has recent data (within last 7 days)
                    recent_data = lookback_window[asset].tail(7).dropna()
                    if len(recent_data) > 0:
                        # Additional check: ensure enough data for momentum calculation
                        if len(asset_data) >= momentum_min_data:
                            available_assets.append(asset)
                        
            if len(available_assets) < 3:  # Need at least 3 assets
                continue
                
            print(f"Date: {current_date.date()}, Available assets: {len(available_assets)}")
            
            # Use only available assets for this rebalancing period
            current_returns = lookback_window[available_assets].dropna()
            
            # Ensure we have enough observations for both momentum and HRP
            min_observations = max(momentum_min_data, int(lookback_days * 0.6))
            if len(current_returns) < min_observations:
                continue
            
            # Calculate momentum scores for available assets with proper validation
            try:
                momentum_scores = current_returns.rolling(momentum_window, min_periods=momentum_window).mean() * np.sqrt(252)
                latest_momentum = momentum_scores.iloc[-1].dropna()
                
                # Additional validation: ensure momentum scores are valid
                latest_momentum = latest_momentum[latest_momentum.notna() & (latest_momentum != 0)]
                
            except Exception as e:
                print(f"Error calculating momentum on {current_date.date()}: {e}")
                continue
            
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
            
            # Apply HRP to the momentum-selected assets
            try:
                # Use HRP to get optimal weights
                hrp_weights = get_hrp_weights(
                    cov=cov, 
                    corr=corr,
                    returns_data=momentum_returns,
                    min_allocation=0.0  # No minimum allocation constraint for the hybrid
                )
                
                # Step 3: Blend HRP weights with momentum weights
                # First, normalize momentum scores to portfolio weights
                momentum_weights = pd.Series(0.0, index=top_momentum_assets, dtype=float)
                
                # Use positive momentum scores as weights (higher momentum = higher weight)
                for asset in top_momentum_assets:
                    if asset in latest_momentum:
                        # Ensure momentum score is positive by using rank-based weights
                        momentum_weights[asset] = latest_momentum[asset]
                
                # Make momentum scores positive and normalize to sum to 1
                if momentum_weights.sum() > 0:
                    # Convert to positive values if any negatives exist
                    min_mom = momentum_weights.min()
                    if min_mom < 0:
                        momentum_weights = momentum_weights - min_mom + 0.001  # Shift to positive
                    
                    # Normalize to sum to 1
                    momentum_weights = momentum_weights / momentum_weights.sum()
                else:
                    # Fallback to equal weights if all momentum scores are zero/negative
                    momentum_weights = pd.Series(1.0/len(top_momentum_assets), index=top_momentum_assets)
                
                # Ensure both weight series use the same index
                common_assets = list(set(hrp_weights.index).intersection(set(momentum_weights.index)))
                
                # Initialize blended weights
                blended_weights = pd.Series(0.0, index=common_assets, dtype=float)
                
                # Calculate blended weights only for common assets
                for asset in common_assets:
                    hrp_weight = hrp_weights.get(asset, 0.0)
                    mom_weight = momentum_weights.get(asset, 0.0)
                    blended_weights[asset] = (1 - momentum_weight) * hrp_weight + momentum_weight * mom_weight
                
                # Normalize blended weights to sum to 1
                if blended_weights.sum() > 0:
                    blended_weights = blended_weights / blended_weights.sum()
                else:
                    # Fallback to equal weights
                    blended_weights = pd.Series(1.0/len(common_assets), index=common_assets)
                    
                # Update current weights with the blended weights
                current_weights = pd.Series(0.0, index=returns_data.columns, dtype=float)
                for asset in blended_weights.index:
                    current_weights[asset] = float(blended_weights[asset])
                
            except Exception as e:
                print(f"Error calculating weights: {e}")
                # Fallback to equal weights if HRP fails
                current_weights = pd.Series(0.0, index=returns_data.columns, dtype=float)
                equal_weight = 1.0 / len(top_momentum_assets)
                for asset in top_momentum_assets:
                    current_weights[asset] = equal_weight

            # Store weights at this rebalancing date
            weights_history[current_date] = current_weights.copy()
            
            # Print active positions
            active_positions = sum(current_weights > 0)
            print(f"Active positions: {active_positions}")
        
        # Update portfolio value based on daily returns
        daily_returns = returns_data.loc[current_date]
        # Calculate portfolio return as weighted sum of asset returns
        portfolio_return = (current_weights * daily_returns).sum()
        # Update portfolio value
        portfolio_value *= (1 + portfolio_return)
        portfolio_values[current_date] = portfolio_value
    
    # Convert weights history to DataFrame for easier analysis
    weights_df = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
    
    for date in weights_history:
        weights_df.loc[date] = weights_history[date]
    
    # Forward fill weights to all dates
    weights_df = weights_df.ffill().fillna(0)
    
    return portfolio_values, weights_df

def calculate_performance_metrics(portfolio_values):
    """Calculate key performance metrics."""
    # Initial and final portfolio values
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
    
    print("\n=== Momentum-Weighted HRP Strategy Performance ===")
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
    plt.title('Momentum-Weighted HRP Strategy Portfolio Value', fontsize=14)
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
    momentum_weight = 0.5  # Weight given to momentum scores (0.5 = equal blend)
    
    # Run the momentum-weighted HRP hybrid backtest
    print("\n=== Running Momentum-Weighted HRP Strategy Backtest (Weekly Rebalancing) ===")
    hybrid_values, hybrid_weights = run_momentum_weighted_hrp_backtest(
        returns, 
        lookback_days=lookback_days,
        momentum_window=momentum_window,
        initial_capital=initial_capital, 
        rebalancing_freq='W',
        n_momentum_assets=n_momentum_assets,
        momentum_weight=momentum_weight
    )
    
    # Print performance results
    print_performance_summary(hybrid_values, hybrid_weights)
    
    # Plot portfolio value
    plot_portfolio_value(hybrid_values, save_path='../../images/momentum_weighted_hrp_portfolio.png')
    
    # Save portfolio values for comparison
    hybrid_values.to_csv('../../data/momentum_weighted_hrp_portfolio_values.csv')
    print("\nSaved portfolio values to 'data/momentum_weighted_hrp_portfolio_values.csv'")
