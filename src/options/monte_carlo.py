"""
Monte Carlo Simulation for Options Pricing

Uses Geometric Brownian Motion to simulate stock price paths
and price options through simulation.
"""

import numpy as np
from typing import Dict, List


class MonteCarloSimulator:
    """
    Monte Carlo simulator for option pricing using GBM
    """
    
    def __init__(self, S0: float, K: float, T: float, r: float, 
                 sigma: float, n_simulations: int = 10000, n_steps: int = 252):
        """
        Initialize Monte Carlo simulator
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        K : float
            Strike price
        T : float
            Time to maturity (years)
        r : float
            Risk-free rate
        sigma : float
            Volatility
        n_simulations : int
            Number of price paths to simulate
        n_steps : int
            Number of time steps per simulation (252 = trading days/year)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.dt = T / n_steps  # Time increment
        
        # Storage for simulated paths
        self.paths = None
        
    def generate_paths(self) -> np.ndarray:
        """
        Generate stock price paths using Geometric Brownian Motion
        
        Uses the exact solution:
        S(t+dt) = S(t) * exp((r - σ²/2)dt + σ√dt*Z)
        
        Returns:
        --------
        np.ndarray : Shape (n_simulations, n_steps+1) of price paths
        """
        # Generate random normal variables for all paths at once
        Z = np.random.standard_normal((self.n_simulations, self.n_steps))
        
        # Initialize paths array
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = self.S0
        
        # Calculate drift and diffusion components
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)
        
        # Generate paths using cumulative sum for efficiency
        increments = drift + diffusion * Z
        log_returns = np.cumsum(increments, axis=1)
        paths[:, 1:] = self.S0 * np.exp(log_returns)
        
        self.paths = paths
        return paths
    
    def price_european_option(self, option_type: str = 'call') -> Dict:
        """
        Price European option using Monte Carlo
        
        Algorithm:
        1. Simulate N stock price paths to maturity
        2. Calculate payoff for each path
        3. Take average payoff
        4. Discount to present value
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        dict : Contains price, standard error, and confidence interval
        """
        if self.paths is None:
            self.generate_paths()
        
        # Extract terminal stock prices (at maturity)
        ST = self.paths[:, -1]
        
        # Calculate payoff for each path
        if option_type == 'call':
            payoffs = np.maximum(ST - self.K, 0)
        else:
            payoffs = np.maximum(self.K - ST, 0)
        
        # Discount payoffs to present value
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        # Calculate standard error (uncertainty in MC estimate)
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations) * \
                    np.exp(-self.r * self.T)
        
        # 95% confidence interval: mean ± 1.96 * SE
        conf_interval = 1.96 * std_error
        
        return {
            'price': option_price,
            'std_error': std_error,
            'conf_lower': option_price - conf_interval,
            'conf_upper': option_price + conf_interval
        }
    
    def price_asian_option(self, option_type: str = 'call') -> Dict:
        """
        Price Asian (average price) option
        
        Asian options: Payoff based on AVERAGE stock price over the period
        - Call: max(S_avg - K, 0)
        - Put: max(K - S_avg, 0)
        
        Returns:
        --------
        dict : Price and statistics
        """
        if self.paths is None:
            self.generate_paths()
        
        # Calculate average price for each path
        S_avg = np.mean(self.paths, axis=1)
        
        # Calculate payoff based on average
        if option_type == 'call':
            payoffs = np.maximum(S_avg - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_avg, 0)
        
        # Discount to present value
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations) * \
                    np.exp(-self.r * self.T)
        
        return {
            'price': option_price,
            'std_error': std_error
        }
    
    def calculate_greeks_mc(self, option_type: str = 'call', 
                           bump_size: float = 0.01) -> Dict:
        """
        Calculate Greeks using finite difference method
        
        Method: Bump and reprice
        - Delta: (V(S+h) - V(S-h)) / (2h)
        - Gamma: (V(S+h) - 2V(S) + V(S-h)) / h²
        - Vega: (V(σ+h) - V(σ-h)) / (2h)
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        bump_size : float
            Size of perturbation for finite differences
            
        Returns:
        --------
        dict : Delta, Gamma, Vega estimates
        """
        # Base price
        V0 = self.price_european_option(option_type)['price']
        
        # Delta: bump stock price
        original_S0 = self.S0
        
        self.S0 = original_S0 + bump_size
        self.paths = None  # Reset paths
        V_up = self.price_european_option(option_type)['price']
        
        self.S0 = original_S0 - bump_size
        self.paths = None
        V_down = self.price_european_option(option_type)['price']
        
        delta = (V_up - V_down) / (2 * bump_size)
        gamma = (V_up - 2*V0 + V_down) / (bump_size**2)
        
        # Reset S0
        self.S0 = original_S0
        
        # Vega: bump volatility
        original_sigma = self.sigma
        
        self.sigma = original_sigma + bump_size
        self.paths = None
        V_sigma_up = self.price_european_option(option_type)['price']
        
        self.sigma = original_sigma - bump_size
        self.paths = None
        V_sigma_down = self.price_european_option(option_type)['price']
        
        vega = (V_sigma_up - V_sigma_down) / (2 * bump_size)
        
        # Reset sigma
        self.sigma = original_sigma
        self.paths = None  # Reset paths for future use
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100  # Per 1% change
        }
    
    def get_sample_paths(self, n_paths: int = 100) -> np.ndarray:
        """
        Get sample paths for visualization
        
        Parameters:
        -----------
        n_paths : int
            Number of paths to return
            
        Returns:
        --------
        np.ndarray : Sample paths
        """
        if self.paths is None:
            self.generate_paths()
        
        indices = np.random.choice(self.n_simulations, 
                                   size=min(n_paths, self.n_simulations), 
                                   replace=False)
        return self.paths[indices]


# Example usage and validation
if __name__ == "__main__":
    print("=" * 60)
    print("Monte Carlo Option Pricing")
    print("=" * 60)
    
    # Parameters
    S0 = 100      # Stock price
    K = 100       # Strike price
    T = 1         # 1 year
    r = 0.05      # 5% risk-free rate
    sigma = 0.20  # 20% volatility
    
    # Initialize simulator
    mc = MonteCarloSimulator(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        n_simulations=100000,
        n_steps=252
    )
    
    # Price European call
    print("\nEuropean Call Option:")
    call_result = mc.price_european_option('call')
    print(f"  MC Price: ${call_result['price']:.4f}")
    print(f"  Std Error: ${call_result['std_error']:.4f}")
    print(f"  95% CI: [${call_result['conf_lower']:.4f}, "
          f"${call_result['conf_upper']:.4f}]")
    
    # Compare with Black-Scholes
    try:
        from src.options.black_scholes import BlackScholesModel
        bs = BlackScholesModel(S0, K, T, r, sigma)
        bs_price = bs.call_price()
        print(f"  BS Price: ${bs_price:.4f}")
        print(f"  Difference: ${abs(call_result['price'] - bs_price):.4f}")
    except ImportError:
        print("  (Black-Scholes comparison unavailable)")
    
    # Price European put
    print("\nEuropean Put Option:")
    put_result = mc.price_european_option('put')
    print(f"  MC Price: ${put_result['price']:.4f}")
    print(f"  Std Error: ${put_result['std_error']:.4f}")
    
    # Calculate Greeks
    print("\nGreeks (Monte Carlo):")
    greeks_mc = mc.calculate_greeks_mc('call')
    print(f"  Delta: {greeks_mc['delta']:.4f}")
    print(f"  Gamma: {greeks_mc['gamma']:.4f}")
    print(f"  Vega: {greeks_mc['vega']:.4f}")
    
    print("\n✅ Monte Carlo simulation complete!")