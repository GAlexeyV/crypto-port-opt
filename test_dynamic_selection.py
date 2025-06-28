#!/usr/bin/env python3
"""
Test Dynamic Asset Selection

This script tests the strategies with proper dynamic asset selection:
1. Load ALL symbols from the dataset (not just clean 10)
2. Apply dynamic filtering at each rebalancing date
3. Compare results
"""

from src.strategies.hrp import run_hrp_backtest
from src.strategies.momentum_weighted_hrp import run_momentum_weighted_hrp_backtest
from src.utils.data_loader import load_returns_data

def test_dynamic_selection():
    """Test strategies with proper dynamic asset selection"""
    
    print("=== Testing Dynamic Asset Selection ===\n")
    
    # Load ALL symbols from dataset (not just clean 10)
    print("1. Loading ALL symbols from dataset...")
    config_path = 'config/strategy_params.yaml'
    
    # Temporarily modify config to use all symbols
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Save original setting
    original_use_all = config['symbols']['use_all_from_data']
    
    # Set to use all symbols
    config['symbols']['use_all_from_data'] = True
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Load all symbols
        returns_all = load_returns_data('data/daily_returns.csv', config_path)
        print(f"Loaded: {len(returns_all)} days, {len(returns_all.columns)} assets")
        print(f"Assets: {list(returns_all.columns)[:10]}... (showing first 10)")
        
        # Check data quality
        print(f"\nData quality check:")
        missing_pct = returns_all.isnull().sum() / len(returns_all) * 100
        missing_pct = missing_pct.sort_values(ascending=False)
        
        print(f"Assets with >50% missing data: {(missing_pct > 50).sum()}")
        print(f"Assets with >20% missing data: {(missing_pct > 20).sum()}")
        print(f"Clean assets (0% missing): {(missing_pct == 0).sum()}")
        
        # Show worst assets
        print(f"\nWorst assets (missing data %):")
        for asset, pct in missing_pct.head().items():
            print(f"  {asset}: {pct:.1f}% missing")
        
        # Test HRP with all symbols
        print(f"\n=== Testing HRP with All Symbols (Dynamic Selection) ===")
        portfolio_values_hrp, weights_history_hrp = run_hrp_backtest(
            returns_all, lookback_days=60, rebalancing_freq='W', initial_capital=100000
        )
        
        hrp_return = (portfolio_values_hrp.iloc[-1] / portfolio_values_hrp.iloc[0] - 1) * 100
        print(f"HRP Total Return (All Symbols): {hrp_return:.2f}%")
        
        # Test Momentum-Weighted HRP with all symbols
        print(f"\n=== Testing Momentum-Weighted HRP with All Symbols ===")
        portfolio_values_mw, weights_history_mw = run_momentum_weighted_hrp_backtest(
            returns_all, lookback_days=60, momentum_window=30, rebalancing_freq='W', 
            initial_capital=100000, n_momentum_assets=20, momentum_weight=0.5
        )
        
        mw_return = (portfolio_values_mw.iloc[-1] / portfolio_values_mw.iloc[0] - 1) * 100
        print(f"Momentum-Weighted HRP Total Return (All Symbols): {mw_return:.2f}%")
        
        # Compare results
        print(f"\n=== Comparison ===")
        print(f"HRP (All Symbols):           {hrp_return:.2f}%")
        print(f"Momentum-Weighted HRP:       {mw_return:.2f}%")
        print(f"Difference:                  {hrp_return - mw_return:.2f} percentage points")
        
    finally:
        # Restore original config
        config['symbols']['use_all_from_data'] = original_use_all
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\nRestored original config: use_all_from_data = {original_use_all}")

if __name__ == "__main__":
    test_dynamic_selection()
