#!/usr/bin/env python3
"""
Main Entry Point for Portfolio Strategy Runner

This script serves as the main entry point for running various portfolio allocation strategies.
It provides a command-line interface to execute different strategies with configurable parameters.

Usage:
    python main.py <strategy> [--config CONFIG_FILE] [--compare-with STRATEGIES...]

Examples:
    python main.py hrp
    python main.py momentum --compare-with hrp btc_hold
    python main.py compare

Available Strategies:
    - hrp: Hierarchical Risk Parity
    - equal_weight: Equal Weight Portfolio
    - btc_hold: Bitcoin Buy and Hold
    - momentum: Momentum Strategy
    - momentum_hrp: Momentum + HRP Hybrid
    - momentum_weighted_hrp: Momentum-Weighted HRP
    - compare: Compare all strategies
"""

import argparse
import warnings
from pathlib import Path

# Import strategies
from src.strategies.hrp import run_hrp_backtest
from src.strategies.equal_weight import run_equal_weight_backtest
from src.strategies.btc_hold import run_btc_hold_backtest
from src.strategies.momentum_strategy import run_momentum_backtest
from src.strategies.momentum_hrp import run_momentum_hrp_backtest
from src.strategies.momentum_weighted_hrp import run_momentum_weighted_hrp_backtest

# Import utility functions
from src.utils.config_loader import get_strategy_params
from src.utils.data_loader import load_returns_data, validate_returns_data


def print_performance_summary(portfolio_values, weights_history=None, initial_capital=100000):
    """Print a summary of strategy performance metrics"""
    # Ensure portfolio values are sorted by date
    portfolio_values = portfolio_values.sort_index()
    
    final_value = portfolio_values.iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")

    # Calculate annualized return
    days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
    annual_return = (final_value / initial_capital) ** (365 / days) - 1 if days > 0 else 0
    print(f"Annualized Return: {annual_return*100:.2f}%")

    # Calculate max drawdown
    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values / rolling_max - 1) * 100
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")

    # Print information about the last set of weights
    if weights_history is not None:
        import pandas as pd
        if isinstance(weights_history, dict):
            if weights_history:
                last_date = max(weights_history.keys())
                last_weights = weights_history[last_date]
                active_positions = (last_weights > 0).sum()
                
                print(f"\nFinal portfolio allocation (date: {last_date})")
                print(f"Number of active positions: {active_positions}")
        elif isinstance(weights_history, pd.DataFrame):
            # Get the last row with non-zero values
            non_zero_rows = weights_history.sum(axis=1)
            non_zero_rows = non_zero_rows[non_zero_rows > 0]
            
            if len(non_zero_rows) > 0:
                last_date = non_zero_rows.index[-1]
                last_weights = weights_history.loc[last_date]
                active_positions = (last_weights > 0).sum()
                
                print(f"\nFinal portfolio allocation (date: {last_date})")
                print(f"Number of active positions: {active_positions}")


def run_strategy(strategy_name, returns, params):
    """
    Run a specific strategy with given parameters.
    
    Parameters
    ----------
    strategy_name : str
        Name of the strategy to run
    returns : pd.DataFrame
        Returns data
    params : dict
        Strategy parameters
        
    Returns
    -------
    tuple
        (portfolio_values, weights_history)
    """
    print(f"\n=== Running {strategy_name.upper()} Strategy Backtest ===")
    
    if strategy_name == 'hrp':
        return run_hrp_backtest(
            returns,
            lookback_days=params.get('lookback_days', 60),
            initial_capital=params.get('initial_capital', 100000),
            rebalancing_freq=params.get('rebalance_frequency', 'W')
        )
    elif strategy_name == 'equal_weight':
        return run_equal_weight_backtest(
            returns,
            lookback_days=params.get('lookback_days', 60),
            initial_capital=params.get('initial_capital', 100000),
            rebalancing_freq=params.get('rebalance_frequency', 'W'),
            n_assets=params.get('n_assets')
        )
    elif strategy_name == 'btc_hold':
        # BTC Hold strategy doesn't need complex parameters, just initial capital
        initial_capital = 100000
        if params and isinstance(params, dict):
            initial_capital = params.get('initial_capital', 100000)
        
        portfolio_values = run_btc_hold_backtest(
            returns,
            initial_capital=initial_capital
        )
        return portfolio_values, None
    elif strategy_name == 'momentum':
        return run_momentum_backtest(
            returns,
            lookback_days=params.get('lookback_days', 60),
            momentum_window=params.get('momentum_window', 30),
            initial_capital=params.get('initial_capital', 100000),
            rebalancing_freq=params.get('rebalance_frequency', 'W'),
            n_assets=params.get('n_assets', 10)
        )
    elif strategy_name == 'momentum_hrp':
        return run_momentum_hrp_backtest(
            returns,
            lookback_days=params.get('lookback_days', 60),
            momentum_window=params.get('momentum_window', 30),
            initial_capital=params.get('initial_capital', 100000),
            rebalancing_freq=params.get('rebalance_frequency', 'W'),
            n_momentum_assets=params.get('n_momentum_assets', 20)
        )
    elif strategy_name == 'momentum_weighted_hrp':
        return run_momentum_weighted_hrp_backtest(
            returns,
            lookback_days=params.get('lookback_days', 60),
            momentum_window=params.get('momentum_window', 30),
            initial_capital=params.get('initial_capital', 100000),
            rebalancing_freq=params.get('rebalance_frequency', 'W'),
            n_momentum_assets=params.get('n_momentum_assets', 20),
            momentum_weight=params.get('momentum_weight', 0.5)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def main():
    """Main function to run strategies from command line"""
    parser = argparse.ArgumentParser(
        description='Run portfolio allocation strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'strategy', 
        choices=['hrp', 'equal_weight', 'btc_hold', 'momentum', 'momentum_hrp', 'momentum_weighted_hrp', 'compare'],
        help='Strategy to run or "compare" to compare multiple strategies'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to configuration file (default: config/strategy_params.yaml)'
    )
    parser.add_argument(
        '--compare-with', 
        nargs='+',
        choices=['hrp', 'equal_weight', 'btc_hold', 'momentum', 'momentum_hrp', 'momentum_weighted_hrp'],
        help='Additional strategies to compare with the main strategy'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Directory to save output files (default: data)'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='images',
        help='Directory to save plots (default: images)'
    )
    
    args = parser.parse_args()
    
    # Set warning filter
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # If we're comparing strategies, run the comparison tool
    if args.strategy == 'compare':
        from src.analysis.strategy_comparison import main as compare_main
        compare_main()
        return
    
    try:
        # Get strategy parameters
        print(f"Loading parameters for strategy: {args.strategy}")
        params = get_strategy_params(args.strategy, args.config)
        print(f"Loaded params: {params}")
        
        # Ensure params is a dictionary
        if params is None:
            params = {}
            print("Params was None, set to empty dict")
            
        # For BTC hold strategy, ensure we have initial capital
        if args.strategy == 'btc_hold' and 'initial_capital' not in params:
            # Set default initial capital from general settings
            general_params = get_strategy_params('general', args.config)
            if general_params and 'initial_capital' in general_params:
                params['initial_capital'] = general_params['initial_capital']
            else:
                params['initial_capital'] = 100000
            print(f"Using initial capital: {params['initial_capital']}")
        
            
        # Load returns data
        data_file = params.get('data_file', 'data/daily_returns.csv')
        print(f"Loading data from: {data_file}")
        returns = load_returns_data(data_file, args.config)
        
        # Validate data
        validation_results = validate_returns_data(returns)
        if validation_results['warnings']:
            print("\nData validation warnings:")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")
        
        # Run the selected strategy
        print(f"Running strategy: {args.strategy} with params: {params}")
        portfolio_values, weights = run_strategy(args.strategy, returns, params)
        
        # Print performance summary
        initial_capital = params.get('initial_capital', 100000)
        print_performance_summary(portfolio_values, weights, initial_capital)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{args.strategy}_portfolio_values.csv"
        portfolio_values.to_csv(output_file)
        print(f"\nSaved results to {output_file}")
        
        # Save plot
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_values.index, portfolio_values, label=f'{args.strategy.upper()} Strategy')
            plt.title(f'{args.strategy.upper()} Strategy Portfolio Value')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            
            plot_file = plot_dir / f"{args.strategy}_portfolio.png"
            plt.savefig(plot_file, dpi=300)
            print(f"Saved plot to {plot_file}")
            plt.show()
            
        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")
        
        # Compare with other strategies if requested
        if args.compare_with:
            try:
                from src.analysis.strategy_comparison import load_strategy_data, calculate_metrics, plot_portfolio_comparison
                
                strategies_data = {args.strategy: portfolio_values}
                for strategy in args.compare_with:
                    # Load the comparison strategy data from file
                    strategy_file = output_dir / f"{strategy}_portfolio_values.csv"
                    if strategy_file.exists():
                        data = load_strategy_data(str(strategy_file), strategy)
                        if data is not None:
                            strategies_data[strategy] = data
                    else:
                        print(f"Warning: Could not find data for {strategy} at {strategy_file}")
                
                # Plot comparison
                if len(strategies_data) > 1:
                    print(f"\nComparing {args.strategy} with: {', '.join(args.compare_with)}")
                    comparison_plot = plot_dir / f"{args.strategy}_comparison.png"
                    plot_portfolio_comparison(strategies_data, save_path=str(comparison_plot))
                    
                    # Calculate and display metrics
                    metrics = {name: calculate_metrics(values) for name, values in strategies_data.items()}
                    
                    print("\n=== Performance Metrics Comparison ===\n")
                    for name, m in metrics.items():
                        print(f"{name}:")
                        print(f"  Annualized Return: {m['annualized_return']*100:.2f}%")
                        print(f"  Volatility: {m['volatility']*100:.2f}%")
                        print(f"  Max Drawdown: {m['max_drawdown']*100:.2f}%")
                        print(f"  Sharpe Ratio: {m['sharpe_ratio']:.2f}")
                        print()
            except ImportError as e:
                print(f"Warning: Could not import comparison tools: {e}")
        
    except Exception as e:
        print(f"Error running strategy: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
