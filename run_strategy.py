#!/usr/bin/env python3
"""
Run Strategy Script

This is a convenience script to run different trading strategies from the command line.
"""

import argparse
import pandas as pd
from src.strategies.momentum_hrp import run_momentum_hrp_backtest
from src.strategies.equal_weight import run_equal_weight_backtest
from src.strategies.btc_hold import run_btc_hold_backtest
from src.strategies.momentum_strategy import run_momentum_backtest
from src.strategies.momentum_weighted_hrp import run_momentum_weighted_hrp_backtest
from src.strategies.hrp import run_hrp_backtest
from src.utils.config_loader import get_strategy_params


def main():
    parser = argparse.ArgumentParser(description='Run a trading strategy backtest')
    parser.add_argument('strategy', choices=['hrp', 'equal_weight', 'btc_hold', 'momentum', 'momentum_hrp', 'momentum_weighted_hrp'],
                        help='Trading strategy to backtest')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (default: config/strategy_params.yaml)')
    parser.add_argument('--initial-capital', type=float,
                        help='Override initial capital from config')
    parser.add_argument('--rebalance-freq', type=str, choices=['W', 'M'],
                        help='Override rebalancing frequency: W for weekly, M for monthly')
    parser.add_argument('--lookback', type=int,
                        help='Override lookback period in days')
    parser.add_argument('--momentum-window', type=int,
                        help='Override momentum calculation window in days')
    parser.add_argument('--n-assets', type=int,
                        help='Override number of assets to include')
    parser.add_argument('--data-file', type=str,
                        help='Override path to returns data CSV file')
    
    args = parser.parse_args()
    
    # Load strategy parameters from config
    params = get_strategy_params(args.strategy, args.config)
    
    # Override config with command line arguments if provided
    if args.initial_capital is not None:
        params['initial_capital'] = args.initial_capital
    if args.rebalance_freq is not None:
        params['rebalance_frequency'] = args.rebalance_freq
    if args.lookback is not None:
        params['lookback_days'] = args.lookback
    if args.momentum_window is not None:
        params['momentum_window'] = args.momentum_window
    if args.n_assets is not None:
        params['n_assets'] = args.n_assets
    if args.data_file is not None:
        params['data_file'] = args.data_file
    
    # Load the returns data
    data_file = params.get('data_file', 'data/daily_returns.csv')
    print(f"Loading returns data from {data_file}")
    returns = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Run the selected strategy
    if args.strategy == 'hrp':
        print("\n=== Running HRP Strategy Backtest ===")
        portfolio_values, weights_history = run_hrp_backtest(
            returns,
            lookback_days=params.get('lookback_days', 60),
            initial_capital=params.get('initial_capital', 100000),
            rebalancing_freq=params.get('rebalance_frequency', 'W')
        )
        output_prefix = 'hrp'
        
    elif args.strategy == 'equal_weight':
        print("\n=== Running Equal Weight Strategy Backtest ===")
        portfolio_values, weights_history = run_equal_weight_backtest(
            returns,
            lookback_days=params.get('lookback_days', 60),
            initial_capital=params.get('initial_capital', 100000),
            rebalancing_freq=params.get('rebalance_frequency', 'W'),
            n_assets=params.get('n_assets')
        )
        output_prefix = 'equal_weight'
        
    elif args.strategy == 'btc_hold':
        print("\n=== Running BTC Hold Strategy Backtest ===")
        portfolio_values = run_btc_hold_backtest(
            returns,
            initial_capital=params.get('initial_capital', 100000)
        )
        # For BTC hold, we don't have weights_history since it's a simple buy-and-hold strategy
        weights_history = None
        output_prefix = 'btc_hold'
        
    elif args.strategy == 'momentum':
        print("\n=== Running Momentum Strategy Backtest ===")
        portfolio_values, weights_history = run_momentum_backtest(
            returns,
            lookback_days=params.get('lookback_days', 60),
            momentum_window=params.get('momentum_window', 30),
            initial_capital=params.get('initial_capital', 100000),
            rebalancing_freq=params.get('rebalance_frequency', 'W'),
            n_assets=params.get('n_assets', 10)
        )
        output_prefix = 'momentum'
        
    elif args.strategy == 'momentum_hrp':
        print("\n=== Running Momentum-HRP Strategy Backtest ===")
        portfolio_values, weights_history = run_momentum_hrp_backtest(
            returns,
            lookback_days=params.get('lookback_days', 60),
            momentum_window=params.get('momentum_window', 30),
            initial_capital=params.get('initial_capital', 100000),
            rebalancing_freq=params.get('rebalance_frequency', 'W'),
            n_momentum_assets=params.get('n_momentum_assets', 20)
        )
        output_prefix = 'momentum_hrp'
        
    elif args.strategy == 'momentum_weighted_hrp':
        print("\n=== Running Momentum-Weighted HRP Strategy Backtest ===")
        portfolio_values, weights_history = run_momentum_weighted_hrp_backtest(
            returns,
            lookback_days=params.get('lookback_days', 60),
            momentum_window=params.get('momentum_window', 30),
            initial_capital=params.get('initial_capital', 100000),
            rebalancing_freq=params.get('rebalance_frequency', 'W'),
            n_momentum_assets=params.get('n_momentum_assets', 20),
            momentum_weight=params.get('momentum_weight', 0.5)
        )
        output_prefix = 'momentum_weighted_hrp'
    
    # Save the results
    csv_file = f'data/{output_prefix}_portfolio_values.csv'
    portfolio_values.to_csv(csv_file)
    print(f"\nSaved portfolio values to '{csv_file}'")
    
    # Print final portfolio value
    initial_value = portfolio_values.iloc[0]
    final_value = portfolio_values.iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    print(f"\nInitial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")


if __name__ == '__main__':
    main()
