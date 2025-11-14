"""
Black-Scholes Option Pricing Model
Developed by Fischer Black and Myron Scholes (1973)

Assumptions:
1. European options (exercise only at maturity)
2. No dividends
3. Constant volatility and risk-free rate
4. Log-normal stock price distribution
5. No transaction costs or taxes
"""

import numpy as np
from scipy.stats import norm
from typing import Dict


class BlackScholesModel:
    """
    Black-Scholes pricing model and Greeks calculator
    
    The model derives option prices by assuming we can create
    a risk-free hedge between the option and underlying stock.
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Initialize Black-Scholes model parameters
        
        Parameters:
        -----------
        S : float
            Current stock price (spot price)
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate (annualized)
        sigma : float
            Volatility (annualized standard deviation of returns)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        # Pre-calculate d1 and d2 (used in all calculations)
        self._calculate_d1_d2()
    
    def _calculate_d1_d2(self):
        """
        Calculate d1 and d2 from the Black-Scholes formula
        
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        d2 = d1 - σ√T
        
        Intuition:
        - d1: Related to probability of option finishing in-the-money
        - d2: Adjusted version accounting for risk-neutral probability
        """
        self.d1 = (np.log(self.S / self.K) + 
                   (self.r + 0.5 * self.sigma**2) * self.T) / \
                  (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
    
    def call_price(self) -> float:
        """
        Calculate European call option price
        
        Formula: C = S*N(d1) - K*e^(-rT)*N(d2)
        
        Interpretation:
        - S*N(d1): Expected benefit from owning the stock
        - K*e^(-rT)*N(d2): Expected cost of exercising option
        
        Returns:
        --------
        float : Call option price
        """
        return (self.S * norm.cdf(self.d1) - 
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
    
    def put_price(self) -> float:
        """
        Calculate European put option price
        
        Formula: P = K*e^(-rT)*N(-d2) - S*N(-d1)
        
        Can also derive from put-call parity:
        P = C - S + K*e^(-rT)
        
        Returns:
        --------
        float : Put option price
        """
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - 
                self.S * norm.cdf(-self.d1))
    
    def delta(self, option_type: str = 'call') -> float:
        """
        Calculate Delta: ∂V/∂S
        
        Measures option price change per $1 change in stock price
        
        Call Delta: N(d1) - ranges from 0 to 1
        Put Delta: N(d1) - 1 - ranges from -1 to 0
        
        Usage: Delta hedging (maintain market-neutral position)
        
        Returns:
        --------
        float : Delta value
        """
        if option_type == 'call':
            return norm.cdf(self.d1)
        else:
            return norm.cdf(self.d1) - 1
    
    def gamma(self) -> float:
        """
        Calculate Gamma: ∂²V/∂S² = ∂Δ/∂S
        
        Measures rate of change of Delta (curvature of option value)
        Same for calls and puts
        
        Highest when at-the-money and near expiration
        
        Usage: Measures hedging risk - high gamma means delta changes fast
        
        Returns:
        --------
        float : Gamma value
        """
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float:
        """
        Calculate Vega: ∂V/∂σ
        
        Measures option price change per 1% change in volatility
        Same for calls and puts
        
        Note: Vega is highest for at-the-money options
        
        Returns:
        --------
        float : Vega value (divided by 100 for 1% moves)
        """
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T) / 100
    
    def theta(self, option_type: str = 'call') -> float:
        """
        Calculate Theta: ∂V/∂t
        
        Measures option value lost per day due to time decay
        Usually negative (options lose value as time passes)
        
        Divided by 365 to get daily theta
        
        Returns:
        --------
        float : Theta value (per day)
        """
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / \
                (2 * np.sqrt(self.T))
        
        if option_type == 'call':
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * \
                    norm.cdf(self.d2)
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * \
                    norm.cdf(-self.d2)
        
        return (term1 + term2) / 365
    
    def rho(self, option_type: str = 'call') -> float:
        """
        Calculate Rho: ∂V/∂r
        
        Measures option price change per 1% change in interest rate
        
        Less important for short-dated options
        More significant for long-dated options (LEAPS)
        
        Returns:
        --------
        float : Rho value (divided by 100 for 1% moves)
        """
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * \
                   norm.cdf(self.d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * \
                   norm.cdf(-self.d2) / 100
    
    def all_greeks(self, option_type: str = 'call') -> Dict:
        """
        Calculate all Greeks at once
        
        Returns:
        --------
        dict : All Greeks values
        """
        price = self.call_price() if option_type == 'call' else self.put_price()
        
        return {
            'price': price,
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(option_type),
            'rho': self.rho(option_type)
        }


# Example usage:
if __name__ == "__main__":
    # Example: Price a call option
    bs = BlackScholesModel(
        S=100,      # Stock price = $100
        K=100,      # Strike price = $100 (at-the-money)
        T=1,        # 1 year to expiration
        r=0.05,     # 5% risk-free rate
        sigma=0.20  # 20% volatility
    )
    
    print("Call Option Analysis:")
    greeks = bs.all_greeks('call')
    for key, value in greeks.items():
        print(f"{key.capitalize()}: {value:.4f}")