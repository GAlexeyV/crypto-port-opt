#!/usr/bin/env python3
"""
Enhanced Strategy Comparison Tool

This script compares selected portfolio allocation strategies with improved visualization:
1. No extra trend lines
2. Customizable strategy selection
3. Cleaner charts with consistent styling

Usage:
    python compare_strategies.py [--strategies STRATEGY1 STRATEGY2 ...]
    
Example:
    python compare_strategies.py --strategies hrp btc_hold
    python compare_strategies.py --strategies all --exclude momentum_hrp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import argparse


def load_strategy_data(file_path, strategy_name):
    """
    Load portfolio values for a specific strategy.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file containing portfolio values
    strategy_name : str
        Name of the strategy to display in charts and tables
        
    Returns
    -------
    pd.Series
        Series of portfolio values indexed by date
    """
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if isinstance(data, pd.DataFrame) and len(data.columns) == 1:
            # If it's a DataFrame with a single column, convert to Series
            return pd.Series(data.iloc[:, 0], name=strategy_name)
        elif isinstance(data, pd.Series):
            data.name = strategy_name
            return data
        else:
            print(f"Warning: Unexpected data format for {strategy_name}")
            return None
    except Exception as e:
        print(f"Error loading {strategy_name} data: {e}")
        return None


def calculate_metrics(portfolio_values):
    """
    Calculate performance metrics for a strategy.
    
    Parameters
    ----------
    portfolio_values : pd.Series
        Series of portfolio values
        
    Returns
    -------
    dict
        Dictionary of performance metrics
    """
    # Ensure we have valid data
    if portfolio_values is None or len(portfolio_values) < 2:
        return {}
    
    # Calculate daily returns
    daily_returns = portfolio_values.pct_change().dropna()
    
    # Basic metrics
    initial_value = portfolio_values.iloc[0]
    final_value = portfolio_values.iloc[-1]
    total_return = final_value / initial_value - 1
    
    # Calculate dates for annualization
    start_date = portfolio_values.index[0]
    end_date = portfolio_values.index[-1]
    years = (end_date - start_date).days / 365.25
    
    # Annualized return
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Calculate drawdowns
    rolling_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min())
    
    # Calculate volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Calculate downside volatility (only negative returns)
    downside_returns = daily_returns.copy()
    downside_returns[downside_returns > 0] = 0
    downside_volatility = downside_returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Calculate Sortino ratio
    sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
    
    # Calculate Calmar ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
    
    return {
        'initial_value': initial_value,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'downside_volatility': downside_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio
    }


def plot_portfolio_comparison(strategies_data, save_path=None):
    """
    Plot portfolio values for all strategies with improved styling.
    
    Parameters
    ----------
    strategies_data : dict
        Dictionary mapping strategy names to Series of portfolio values
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    # Use a consistent color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies_data)))
    
    # Sort strategies by final value for more logical legend ordering
    sorted_strategies = {k: v for k, v in sorted(
        strategies_data.items(),
        key=lambda item: item[1].iloc[-1] if len(item[1]) > 0 else 0,
        reverse=True
    )}
    
    for i, (strategy_name, values) in enumerate(sorted_strategies.items()):
        # Plot each strategy with a distinct color and line style
        plt.plot(
            values.index, 
            values, 
            label=strategy_name,
            color=colors[i],
            linewidth=2,
            alpha=0.9
        )
    
    plt.title('Portfolio Value Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create a more visible legend
    plt.legend(
        loc='upper left',
        framealpha=0.9,
        frameon=True,
        fontsize=10
    )
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x:,.0f}')
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved portfolio comparison chart to {save_path}")
    
    plt.close()  # Close the figure to prevent display in notebooks


def plot_drawdowns(strategies_data, save_path=None):
    """
    Plot drawdowns for all strategies with improved styling.
    
    Parameters
    ----------
    strategies_data : dict
        Dictionary mapping strategy names to Series of portfolio values
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    # Use a consistent color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies_data)))
    
    # Sort strategies by final value for more logical legend ordering
    sorted_strategies = {k: v for k, v in sorted(
        strategies_data.items(),
        key=lambda item: item[1].iloc[-1] if len(item[1]) > 0 else 0,
        reverse=True
    )}
    
    for i, (strategy_name, values) in enumerate(sorted_strategies.items()):
        # Calculate drawdowns
        rolling_max = values.cummax()
        drawdowns = (values - rolling_max) / rolling_max
        
        # Plot each strategy with a distinct color and line style
        plt.plot(
            drawdowns.index, 
            drawdowns, 
            label=strategy_name,
            color=colors[i],
            linewidth=2,
            alpha=0.9
        )
    
    plt.title('Drawdown Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create a more visible legend
    plt.legend(
        loc='lower left',
        framealpha=0.9,
        frameon=True,
        fontsize=10
    )
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.0%}')
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved drawdown comparison chart to {save_path}")
    
    plt.close()  # Close the figure to prevent display in notebooks


def plot_metrics_comparison(metrics, save_path=None):
    """
    Create bar charts for comparing metrics across strategies with improved styling.
    
    Parameters
    ----------
    metrics : dict
        Dictionary mapping strategy names to dictionaries of metrics
    save_path : str, optional
        Path to save the figure
    """
    # Sort strategies by annualized return
    strategies = sorted(metrics.keys(), 
                       key=lambda s: metrics[s]['annualized_return'],
                       reverse=True)
    
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Strategy Performance Metrics Comparison', fontsize=16)
    
    # Use a consistent color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    # Plot 1: Returns
    returns = [metrics[s]['annualized_return'] * 100 for s in strategies]
    axs[0, 0].bar(strategies, returns, color=colors)
    axs[0, 0].set_title('Annualized Return (%)', fontsize=14)
    axs[0, 0].set_ylabel('Percent (%)', fontsize=12)
    for i, v in enumerate(returns):
        axs[0, 0].text(i, v + 0.5, f"{v:.1f}%", ha='center')
    axs[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Volatility
    vols = [metrics[s]['volatility'] * 100 for s in strategies]
    axs[0, 1].bar(strategies, vols, color=colors)
    axs[0, 1].set_title('Annualized Volatility (%)', fontsize=14)
    axs[0, 1].set_ylabel('Percent (%)', fontsize=12)
    for i, v in enumerate(vols):
        axs[0, 1].text(i, v + 0.5, f"{v:.1f}%", ha='center')
    axs[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Max Drawdown
    drawdowns = [metrics[s]['max_drawdown'] * 100 for s in strategies]
    axs[1, 0].bar(strategies, drawdowns, color=colors)
    axs[1, 0].set_title('Maximum Drawdown (%)', fontsize=14)
    axs[1, 0].set_ylabel('Percent (%)', fontsize=12)
    for i, v in enumerate(drawdowns):
        axs[1, 0].text(i, v + 0.5, f"{v:.1f}%", ha='center')
    axs[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Sharpe Ratio
    sharpes = [metrics[s]['sharpe_ratio'] for s in strategies]
    axs[1, 1].bar(strategies, sharpes, color=colors)
    axs[1, 1].set_title('Sharpe Ratio', fontsize=14)
    for i, v in enumerate(sharpes):
        axs[1, 1].text(i, v + 0.05, f"{v:.2f}", ha='center')
    axs[1, 1].grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    for ax in axs.flat:
        ax.tick_params(axis='x', labelrotation=45)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved metrics comparison chart to {save_path}")
    
    plt.close()  # Close the figure to prevent display in notebooks


def print_metrics_table(metrics):
    """
    Print a formatted table of metrics for all strategies.
    
    Parameters
    ----------
    metrics : dict
        Dictionary mapping strategy names to dictionaries of metrics
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the metrics
    """
    # Create a DataFrame with all metrics
    metrics_df = pd.DataFrame({
        'Strategy': [],
        'Total Return (%)': [],
        'Annualized Return (%)': [],
        'Volatility (%)': [],
        'Max Drawdown (%)': [],
        'Sharpe Ratio': [],
        'Sortino Ratio': [],
        'Calmar Ratio': []
    })
    
    for strategy, m in metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Strategy': [strategy],
            'Total Return (%)': [m['total_return'] * 100],
            'Annualized Return (%)': [m['annualized_return'] * 100],
            'Volatility (%)': [m['volatility'] * 100],
            'Max Drawdown (%)': [m['max_drawdown'] * 100],
            'Sharpe Ratio': [m['sharpe_ratio']],
            'Sortino Ratio': [m['sortino_ratio']],
            'Calmar Ratio': [m['calmar_ratio']]
        })], ignore_index=True)
    
    # Sort by annualized return
    metrics_df = metrics_df.sort_values('Annualized Return (%)', ascending=False)
    
    print("\n=== Strategy Performance Metrics ===\n")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    return metrics_df


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(description='Compare portfolio strategies')
    parser.add_argument('--strategies', nargs='+', 
                        default=['all'], 
                        help='List of strategies to compare, or "all" for all strategies')
    parser.add_argument('--exclude', nargs='+', default=[],
                        help='List of strategies to exclude from comparison')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing strategy data files')
    parser.add_argument('--output-dir', type=str, default='images',
                        help='Directory to save output images')
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots in addition to saving them')
    args = parser.parse_args()
    
    # Define available strategies and their data files
    available_strategies = {
        'hrp': ('hrp_portfolio_values.csv', 'HRP'),
        'equal_weight': ('equal_weight_portfolio_values.csv', 'Equal Weight'),
        'momentum': ('momentum_portfolio_values.csv', 'Momentum'),
        'momentum_hrp': ('momentum_hrp_portfolio_values.csv', 'Momentum-HRP'),
        'momentum_weighted_hrp': ('momentum_weighted_hrp_portfolio_values.csv', 'Momentum-Weighted HRP'),
        'btc_hold': ('btc_hold_portfolio_values.csv', 'BTC Hold')
    }
    
    # Determine which strategies to include
    if 'all' in args.strategies:
        strategies_to_include = available_strategies.keys()
    else:
        strategies_to_include = [s for s in args.strategies if s in available_strategies]
    
    # Apply exclusions
    strategies_to_include = [s for s in strategies_to_include if s not in args.exclude]
    
    if not strategies_to_include:
        print("No valid strategies selected. Exiting.")
        return
    
    # Ensure data and output directories exist
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load strategy data
    print("Loading strategy data...")
    strategies_data = {}
    for strategy in strategies_to_include:
        if strategy in available_strategies:
            file_name, display_name = available_strategies[strategy]
            file_path = data_dir / file_name
            
            if not file_path.exists():
                print(f"Warning: Data file not found for {strategy} at {file_path}")
                continue
                
            data = load_strategy_data(str(file_path), display_name)
            if data is not None:
                strategies_data[display_name] = data
                print(f"Loaded {display_name} strategy data")
    
    if not strategies_data:
        print("No strategy data loaded. Exiting.")
        return
    
    # Calculate metrics for each strategy
    print("\nCalculating performance metrics...")
    metrics = {}
    for strategy_name, values in strategies_data.items():
        metrics[strategy_name] = calculate_metrics(values)
    
    # Print metrics table
    metrics_df = print_metrics_table(metrics)
    
    # Save metrics to CSV
    metrics_csv = data_dir / 'strategy_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics to {metrics_csv}")
    
    # Plot portfolio value comparison
    print("\nPlotting portfolio values...")
    plot_portfolio_comparison(strategies_data, save_path=output_dir / 'portfolio_comparison.png')
    
    # Plot drawdowns
    print("\nPlotting drawdowns...")
    plot_drawdowns(strategies_data, save_path=output_dir / 'drawdown_comparison.png')
    
    # Plot metrics comparison
    print("\nPlotting metrics comparison...")
    plot_metrics_comparison(metrics, save_path=output_dir / 'metrics_comparison.png')
    
    # Show plots if requested
    if args.show_plots:
        import matplotlib.pyplot as plt
        
        # Show portfolio comparison
        plot_portfolio_comparison(strategies_data)
        plt.show()
        
        # Show drawdown comparison
        plot_drawdowns(strategies_data)
        plt.show()
        
        # Show metrics comparison
        plot_metrics_comparison(metrics)
        plt.show()
    
    print("\nAnalysis complete. Results saved to:")
    print(f"  - Performance metrics: {metrics_csv}")
    print(f"  - Portfolio comparison chart: {output_dir / 'portfolio_comparison.png'}")
    print(f"  - Drawdown comparison chart: {output_dir / 'drawdown_comparison.png'}")
    print(f"  - Metrics comparison chart: {output_dir / 'metrics_comparison.png'}")


if __name__ == "__main__":
    main()
