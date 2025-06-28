"""
Bitcoin Buy and Hold Strategy

This script implements a simple buy-and-hold strategy for Bitcoin over the same
time period as our other strategies for fair comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import itertools

def run_btc_hold_backtest(returns_data, initial_capital=100000):
    """
    Run a backtest on a simple Bitcoin buy and hold strategy.
    
    Parameters:
    - returns_data: DataFrame with asset returns
    - initial_capital: Initial capital to invest
    
    Returns:
    - portfolio_values: Series of portfolio values over time
    """
    # Ensure BTC returns are present in the data
    if 'BTCUSDT' not in returns_data.columns:
        raise ValueError("BTCUSDT not found in returns data")
    
    # Get BTC returns
    btc_returns = returns_data['BTCUSDT']
    
    # Calculate cumulative returns
    btc_cumulative_returns = (1 + btc_returns).cumprod()
    
    # Calculate portfolio values
    portfolio_values = initial_capital * btc_cumulative_returns
    
    return portfolio_values

def calculate_performance_metrics(portfolio_values):
    """Calculate comprehensive performance metrics."""
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
    
    # Sortino ratio (downside risk only)
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252)
    sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
    
    # Calmar ratio (return / max drawdown)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Maximum consecutive days down
    down_days = (returns < 0).astype(int)
    max_consecutive_down = max(sum(1 for _ in group) for key, group in itertools.groupby(down_days) if key == 1)
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_consecutive_down': max_consecutive_down
    }

def print_performance_summary(portfolio_values):
    """Print performance metrics summary."""
    metrics = calculate_performance_metrics(portfolio_values)
    
    print("\n=== Bitcoin Buy and Hold Strategy Performance ===")
    print(f"Initial Capital: ${portfolio_values.iloc[0]:,.2f}")
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Volatility (Annualized): {metrics['volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"Max Consecutive Down Days: {metrics['max_consecutive_down']}")

def plot_portfolio_value(portfolio_values, save_path=None):
    """Plot portfolio value over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values.index, portfolio_values, label='Portfolio Value')
    plt.title('Bitcoin Buy and Hold Strategy Portfolio Value', fontsize=14)
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
    
    # Run the BTC buy and hold backtest
    print("\n=== Running Bitcoin Buy and Hold Strategy Backtest ===")
    btc_values = run_btc_hold_backtest(returns, initial_capital=100000)
    
    # Print performance results
    print_performance_summary(btc_values)
    
    # Plot portfolio value
    plot_portfolio_value(btc_values, save_path='../../images/btc_hold_portfolio.png')
    
    # Save portfolio values for comparison
    btc_values.to_csv('../../data/btc_hold_portfolio_values.csv')
    print("\nSaved portfolio values to 'data/btc_hold_portfolio_values.csv'")
