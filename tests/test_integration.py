"""Integration test - Full workflow"""

import sys
sys.path.append('src')

from portfolio.optimizer import PortfolioOptimizer
from options.black_scholes import BlackScholesModel
from options.monte_carlo import MonteCarloSimulator
from utils.data_loader import MarketDataLoader

def test_full_workflow():
    print("=" * 50)
    print("QUANTITATIVE FINANCE TOOLKIT - INTEGRATION TEST")
    print("=" * 50)
    
    # 1. Load real market data
    print("\n1. Loading Market Data...")
    loader = MarketDataLoader()
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    prices = loader.fetch_stock_data(tickers, start_date='2023-01-01')
    returns = loader.calculate_returns(prices)
    print(f"✓ Loaded {len(prices)} days of data for {len(tickers)} stocks")
    
    # 2. Optimize portfolio
    print("\n2. Running Portfolio Optimization...")
    optimizer = PortfolioOptimizer(returns)
    result = optimizer.optimize(target_return=0.15)
    print(f"✓ Optimal weights: {result['weights']}")
    print(f"  Expected Return: {result['return']:.2%}")
    print(f"  Volatility: {result['volatility']:.2%}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    
    # 3. Price options with Black-Scholes
    print("\n3. Pricing Options (Black-Scholes)...")
    current_price = prices['AAPL'].iloc[-1]
    bs = BlackScholesModel(S=current_price, K=current_price, T=0.25, 
                          r=0.05, sigma=0.30)
    call_price = bs.call_price()
    greeks = bs.all_greeks('call')
    print(f"✓ AAPL Call Option (ATM, 3 months):")
    print(f"  Price: ${call_price:.2f}")
    print(f"  Delta: {greeks['delta']:.4f}")
    print(f"  Gamma: {greeks['gamma']:.4f}")
    
    # 4. Price with Monte Carlo
    print("\n4. Pricing Options (Monte Carlo)...")
    mc = MonteCarloSimulator(S0=current_price, K=current_price, T=0.25,
                            r=0.05, sigma=0.30, n_simulations=50000)
    mc_result = mc.price_european_call()
    print(f"✓ Monte Carlo Call Price: ${mc_result['price']:.2f}")
    print(f"  95% CI: ±${mc_result['confidence_interval']:.2f}")
    print(f"  Difference from BS: ${abs(call_price - mc_result['price']):.2f}")
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)

if __name__ == "__main__":
    test_full_workflow()