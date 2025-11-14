"""
Monte Carlo Simulation for Option Pricing
Alternative to Black-Scholes for complex derivatives

Advantages:
- Can price path-dependent options (Asian, Barrier, Lookback)
- Handles multiple underlying assets
- Works with any stochastic process

Method:
1. Simulate many possible stock price paths
2. Calculate payoff for each path
3. Average the payoffs and discount to present value
"""

import numpy as np
from typing import Dict, Callable


class MonteCarloSimulator:
    """
    Monte Carlo option pricing using Geometric Brownian Motion
    
    Stock price follows: dS = μS*dt + σS*dW
    Where dW is a Wiener process (random walk)
    """
    
    def __init__(self, S0: float, K: float, T: float, r: float, 
                 sigma: float, n_simulations: int = 100000, n_steps: int = 252):
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
            Time steps per simulation (252 = daily for 1 year)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.dt = T / n_steps
        
    def simulate_paths(self) -> np.ndarray:
        """
        Simulate stock price paths using Geometric Brownian Motion
        
        Formula: S(t+dt) = S(t) * exp((r - σ²/2)*dt + σ*√dt*Z)
        Where Z ~ N(0,1)
        
        Returns:
        --------
        np.ndarray : Shape (n_simulations, n_steps+1)
        """
        # Initialize price paths
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = self.S0
        
        # Generate random shocks
        Z = np.random.standard_normal((self.n_simulations, self.n_steps))
        
        # Simulate paths
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt + 
                self.sigma * np.sqrt(self.dt) * Z[:, t-1]
            )
        
        return paths
    
    def price_european_call(self) -> Dict:
        """
        Price European call option using Monte Carlo
        
        Returns:
        --------
        dict : Price, standard error, and confidence interval
        """
        paths = self.simulate_paths()
        final_prices = paths[:, -1]
        
        # Calculate payoffs
        payoffs = np.maximum(final_prices - self.K, 0)
        
        # Discount to present value
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        # Calculate standard error
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        confidence_interval = 1.96 * std_error * np.exp(-self.r * self.T)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': confidence_interval
        }
    
    def price_european_put(self) -> Dict:
        """Price European put option"""
        paths = self.simulate_paths()
        final_prices = paths[:, -1]
        
        payoffs = np.maximum(self.K - final_prices, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        confidence_interval = 1.96 * std_error * np.exp(-self.r * self.T)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': confidence_interval
        }
    
    def price_asian_call(self) -> Dict:
        """
        Price Asian call option (payoff based on average price)
        
        Payoff = max(Average(S) - K, 0)
        """
        paths = self.simulate_paths()
        average_prices = np.mean(paths, axis=1)
        
        payoffs = np.maximum(average_prices - self.K, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        confidence_interval = 1.96 * std_error * np.exp(-self.r * self.T)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': confidence_interval
        }
    
    def price_barrier_option(self, barrier: float, option_type: str = 'up-and-out-call') -> Dict:
        """
        Price barrier options (knocked out if price hits barrier)
        
        Parameters:
        -----------
        barrier : float
            Barrier level
        option_type : str
            'up-and-out-call', 'down-and-out-put', etc.
        """
        paths = self.simulate_paths()
        final_prices = paths[:, -1]
        
        if option_type == 'up-and-out-call':
            # Option is knocked out if price goes above barrier
            knockout = np.any(paths >= barrier, axis=1)
            payoffs = np.maximum(final_prices - self.K, 0)
            payoffs[knockout] = 0
        elif option_type == 'down-and-out-put':
            # Option is knocked out if price goes below barrier
            knockout = np.any(paths <= barrier, axis=1)
            payoffs = np.maximum(self.K - final_prices, 0)
            payoffs[knockout] = 0
        
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'knockout_probability': np.mean(knockout)
        }


if __name__ == "__main__":
    # Compare Monte Carlo with Black-Scholes
    mc = MonteCarloSimulator(
        S0=100, K=100, T=1, r=0.05, sigma=0.20,
        n_simulations=100000
    )
    
    call_result = mc.price_european_call()
    print(f"European Call Price: ${call_result['price']:.4f}")
    print(f"95% CI: ±${call_result['confidence_interval']:.4f}")
    
    # Price exotic option
    asian_result = mc.price_asian_call()
    print(f"\nAsian Call Price: ${asian_result['price']:.4f}")