#!/usr/bin/env python3
"""
Run All Strategies and Compare Results

This script automates running all trading strategies and then compares their results
using the enhanced comparison tool that removes extra trend lines.

It will:
1. Run each strategy one by one
2. Generate performance metrics for each strategy
3. Create clean comparison charts without extra trend lines
4. Save results to the data/ and images/ directories
"""

import subprocess
from pathlib import Path
import time
import argparse

def run_command(command):
    """Run a command and print its output in real-time"""
    print(f"\n>>> Running: {' '.join(command)}\n")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run all strategies and compare results')
    parser.add_argument('--strategies', nargs='+', default=['all'],
                        help='List of strategies to run, or "all" for all strategies')
    parser.add_argument('--exclude', nargs='+', default=[],
                        help='List of strategies to exclude')
    parser.add_argument('--only-compare', action='store_true',
                        help='Skip running strategies, only generate comparison')
    parser.add_argument('--show-plots', action='store_true',
                        help='Show plots in addition to saving them')
    args = parser.parse_args()
    
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("images").mkdir(exist_ok=True)
    
    # List of all available strategies
    all_strategies = [
        "hrp",
        "equal_weight",
        "btc_hold",
        "momentum",
        "momentum_hrp",
        "momentum_weighted_hrp"
    ]
    
    # Determine which strategies to run
    if 'all' in args.strategies:
        strategies_to_run = [s for s in all_strategies if s not in args.exclude]
    else:
        strategies_to_run = [s for s in args.strategies if s in all_strategies and s not in args.exclude]
    
    if not strategies_to_run and not args.only_compare:
        print("No valid strategies selected to run. Exiting.")
        return
    
    if not args.only_compare:
        print("=" * 80)
        print("RUNNING SELECTED STRATEGIES")
        print("=" * 80)
        
        # Run each strategy
        for strategy in strategies_to_run:
            print(f"\n{'=' * 20} Running {strategy.upper()} Strategy {'=' * 20}")
            success = run_command(["python", "main.py", strategy])
            if not success:
                print(f"Warning: Failed to run {strategy} strategy")
    
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON REPORTS")
    print("=" * 80)
    
    # Build command for comparison script
    compare_cmd = ["python", "compare_strategies.py"]
    
    # Add strategies argument if specific ones were requested
    if 'all' not in args.strategies:
        compare_cmd.extend(["--strategies"] + args.strategies)
    
    # Add exclude argument if any strategies should be excluded
    if args.exclude:
        compare_cmd.extend(["--exclude"] + args.exclude)
    
    # Add show-plots flag if requested
    if args.show_plots:
        compare_cmd.append("--show-plots")
    
    # Run the comparison script
    run_command(compare_cmd)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Results saved to:")
    print("  - Portfolio values: data/*_portfolio_values.csv")
    print("  - Performance metrics: data/strategy_metrics.csv")
    print("  - Portfolio comparison chart: images/portfolio_comparison.png")
    print("  - Drawdown comparison chart: images/drawdown_comparison.png")
    print("  - Metrics comparison chart: images/metrics_comparison.png")

if __name__ == "__main__":
    start_time = time.time()
    main()
    duration = time.time() - start_time
    print(f"\nTotal execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
