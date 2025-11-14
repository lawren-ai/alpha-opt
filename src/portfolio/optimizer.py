"""
Portfolio Optimizer using scipy.optimize (Windows-friendly alternative to cvxpy)
Based on Harry Markowitz's Modern Portfolio Theory (1952)

This version uses Sequential Least Squares Programming (SLSQP)
instead of cvxpy for better Windows compatibility.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Tuple


class PortfolioOptimizer:
    """
    Implements mean-variance portfolio optimization using scipy
    
    Theory:
    -------
    We want to find portfolio weights that minimize risk (variance)
    while achieving a target return. This is a quadratic programming problem.
    
    minimize:    σ²(w) = w^T * Σ * w
    subject to:  w^T * μ = r_target
                 Σw_i = 1
                 w ≥ 0
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize optimizer with historical returns
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns for each asset (columns = assets, rows = time periods)
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        
        # Calculate expected returns (mean of historical returns)
        # Annualize by multiplying by 252 trading days
        self.expected_returns = returns.mean().values * 252
        
        # Calculate covariance matrix
        # Annualize by multiplying by 252
        self.cov_matrix = returns.cov().values * 252
        
    def portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
            
        Returns:
        --------
        tuple : (return, volatility, sharpe_ratio)
        """
        # Portfolio return: w^T * μ
        portfolio_return = np.dot(weights, self.expected_returns)
        
        # Portfolio variance: w^T * Σ * w
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        # Portfolio volatility (standard deviation)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility (objective function to minimize)
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
            
        Returns:
        --------
        float : Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def optimize(self, target_return: float = None, 
                 risk_free_rate: float = 0.02) -> Dict:
        """
        Find optimal portfolio weights using scipy.optimize
        
        Parameters:
        -----------
        target_return : float
            Desired portfolio return (annualized)
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation
            
        Returns:
        --------
        dict : Contains weights, return, volatility, Sharpe ratio
        """
        
        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Bounds: weights between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Constraint 1: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Constraint 2: achieve target return (if specified)
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, self.expected_returns) - target_return
            })
        
        # Minimize portfolio volatility
        result = minimize(
            fun=self.portfolio_volatility,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Calculate portfolio statistics
        portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_stats(optimal_weights)
        
        return {
            'weights': dict(zip(self.assets, optimal_weights)),
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success
        }
    
    def max_sharpe_portfolio(self, risk_free_rate: float = 0.02) -> Dict:
        """
        Find portfolio with maximum Sharpe ratio (tangency portfolio)
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free rate
            
        Returns:
        --------
        dict : Optimal portfolio with max Sharpe ratio
        """
        
        # Objective: minimize negative Sharpe ratio (maximize Sharpe)
        def neg_sharpe(weights):
            ret, vol, _ = self.portfolio_stats(weights)
            return -(ret - risk_free_rate) / vol
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Constraint: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Optimize
        result = minimize(
            fun=neg_sharpe,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_stats(optimal_weights)
        
        return {
            'weights': dict(zip(self.assets, optimal_weights)),
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success
        }
    
    def min_volatility_portfolio(self) -> Dict:
        """
        Find minimum volatility portfolio (leftmost point on efficient frontier)
        
        Returns:
        --------
        dict : Minimum variance portfolio
        """
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Constraint: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Minimize volatility
        result = minimize(
            fun=self.portfolio_volatility,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_stats(optimal_weights)
        
        return {
            'weights': dict(zip(self.assets, optimal_weights)),
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success
        }
    
    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Generate the efficient frontier
        
        The efficient frontier is the set of optimal portfolios offering
        the highest expected return for each level of risk.
        
        Parameters:
        -----------
        n_points : int
            Number of points to generate along the frontier
            
        Returns:
        --------
        pd.DataFrame : Returns and volatilities of optimal portfolios
        """
        # Find min and max returns for the frontier
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        
        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        results = []
        for target_ret in target_returns:
            try:
                result = self.optimize(target_return=target_ret)
                if result['success']:
                    results.append({
                        'return': result['return'],
                        'volatility': result['volatility'],
                        'sharpe_ratio': result['sharpe_ratio']
                    })
            except:
                continue
                
        return pd.DataFrame(results)


# Example usage:
if __name__ == "__main__":
    # Generate sample returns data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Simulate returns for 4 assets (daily returns)
    returns = pd.DataFrame({
        'Stock_A': np.random.normal(0.0003, 0.02, len(dates)),
        'Stock_B': np.random.normal(0.0002, 0.015, len(dates)),
        'Stock_C': np.random.normal(0.0001, 0.01, len(dates)),
        'Bonds': np.random.normal(0.00005, 0.005, len(dates))
    }, index=dates)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns)
    
    print("\n1. MINIMUM VOLATILITY PORTFOLIO")
    print("-" * 70)
    min_vol = optimizer.min_volatility_portfolio()
    print(f"Expected Return: {min_vol['return']:.2%}")
    print(f"Volatility:      {min_vol['volatility']:.2%}")
    print(f"Sharpe Ratio:    {min_vol['sharpe_ratio']:.2f}")
    print("\nWeights:")
    for asset, weight in min_vol['weights'].items():
        print(f"  {asset}: {weight:.2%}")
    
    print("\n2. MAXIMUM SHARPE RATIO PORTFOLIO")
    print("-" * 70)
    max_sharpe = optimizer.max_sharpe_portfolio()
    print(f"Expected Return: {max_sharpe['return']:.2%}")
    print(f"Volatility:      {max_sharpe['volatility']:.2%}")
    print(f"Sharpe Ratio:    {max_sharpe['sharpe_ratio']:.2f}")
    print("\nWeights:")
    for asset, weight in max_sharpe['weights'].items():
        print(f"  {asset}: {weight:.2%}")
    
    print("\n3. TARGET RETURN PORTFOLIO (10% annual return)")
    print("-" * 70)
    target_portfolio = optimizer.optimize(target_return=0.10)
    print(f"Expected Return: {target_portfolio['return']:.2%}")
    print(f"Volatility:      {target_portfolio['volatility']:.2%}")
    print(f"Sharpe Ratio:    {target_portfolio['sharpe_ratio']:.2f}")
    print("\nWeights:")
    for asset, weight in target_portfolio['weights'].items():
        print(f"  {asset}: {weight:.2%}")
    
    print("\n4. EFFICIENT FRONTIER")
    print("-" * 70)
    frontier = optimizer.efficient_frontier(n_points=20)
    print(f"Generated {len(frontier)} points on the efficient frontier")
    print("\nSample points:")
    print(frontier.head(10).to_string(index=False))
    