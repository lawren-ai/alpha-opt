# 📈 AlphaOPT

A comprehensive Python-based portfolio optimization and options pricing platform with real-time market data integration and interactive visualizations.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.24+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Features

### 📊 Portfolio Optimization
- **Mean-Variance Optimization** based on Modern Portfolio Theory (Markowitz, 1952)
- Interactive efficient frontier visualization
- Sharpe ratio maximization
- Customizable constraints and target returns
- Real-time portfolio rebalancing calculations

### 🎯 Options Pricing
- **Black-Scholes Model** for European options
- Complete Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- **Monte Carlo simulation** for exotic derivatives
- Side-by-side pricing comparison
- Interactive parameter adjustments

### 📉 Market Data Analysis
- Real-time data integration via Yahoo Finance
- Historical price charts and trend analysis
- Returns distribution and volatility calculations
- Multi-asset correlation analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/quant-finance-toolkit.git
cd quant-finance-toolkit

# Create and activate virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## 💻 Usage Examples

### Portfolio Optimization
```python
from src.portfolio.optimizer import PortfolioOptimizer
from src.utils.data_loader import MarketDataLoader

# Fetch market data
loader = MarketDataLoader()
prices = loader.fetch_stock_data(
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN'], 
    start_date='2023-01-01'
)
returns = loader.calculate_returns(prices)

# Optimize portfolio
optimizer = PortfolioOptimizer(returns)
result = optimizer.optimize(target_return=0.15)

print(f"Optimal Weights: {result['weights']}")
print(f"Expected Return: {result['return']:.2%}")
print(f"Volatility: {result['volatility']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
```

### Options Pricing (Black-Scholes)
```python
from src.options.black_scholes import BlackScholesModel

# Initialize model
bs = BlackScholesModel(
    S=150,      # Current stock price
    K=150,      # Strike price
    T=0.25,     # Time to expiration (years)
    r=0.05,     # Risk-free rate
    sigma=0.30  # Volatility
)

# Calculate option price and Greeks
call_price = bs.call_price()
greeks = bs.all_greeks('call')

print(f"Call Option Price: ${call_price:.2f}")
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
```

### Monte Carlo Simulation
```python
from src.options.monte_carlo import MonteCarloSimulator

# Run simulation
mc = MonteCarloSimulator(
    S0=150, K=150, T=0.25, 
    r=0.05, sigma=0.30,
    n_simulations=100000
)

result = mc.price_european_call()
print(f"Option Price: ${result['price']:.2f}")
print(f"95% CI: ±${result['confidence_interval']:.2f}")
```

## 📁 Project Structure
```
alpha-opt/
├── src/
│   ├── portfolio/
│   │   ├── __init__.py
│   │   └── optimizer.py              # Mean-variance optimization
│   ├── options/
│   │   ├── __init__.py
│   │   ├── black_scholes.py          # Analytical pricing model
│   │   └── monte_carlo.py            # Numerical simulation
│   └── utils/
│       ├── __init__.py
│       └── data_loader.py            # Market data fetching
├── tests/
│   └── test_integration.py           # Integration tests
├── app.py                            # Streamlit dashboard
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core programming language |
| **NumPy** | Numerical computations and array operations |
| **Pandas** | Data manipulation and time series |
| **CVXPY** | Convex optimization solver |
| **SciPy** | Statistical distributions and scientific computing |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Dynamic visualizations |
| **yfinance** | Real-time market data API |

## 🎓 Theory & Mathematics

### Modern Portfolio Theory
Implementation of Harry Markowitz's mean-variance framework:

**Optimization Problem:**
```
minimize:    (1/2) * w^T * Σ * w
subject to:  w^T * μ = r_target
             w^T * 1 = 1
             w ≥ 0
```

Where:
- `w` = portfolio weights
- `Σ` = covariance matrix
- `μ` = expected returns vector
- `r_target` = target return

### Black-Scholes Model
Closed-form solution for European options:

**Call Option Price:**
```
C = S*N(d₁) - K*e^(-rT)*N(d₂)
```

**Greeks:**
- **Delta (Δ)**: ∂V/∂S - Price sensitivity to underlying
- **Gamma (Γ)**: ∂²V/∂S² - Delta sensitivity
- **Vega (ν)**: ∂V/∂σ - Volatility sensitivity
- **Theta (Θ)**: ∂V/∂t - Time decay
- **Rho (ρ)**: ∂V/∂r - Interest rate sensitivity

### Monte Carlo Simulation
Simulates stock price paths using Geometric Brownian Motion:
```
S(t+Δt) = S(t) * exp((r - σ²/2)Δt + σ√Δt*Z)
```

Where Z ~ N(0,1)

## 🧪 Testing

Run the integration test suite:
```bash
python tests/test_integration.py
```

## 📊 Screenshots

### Portfolio Optimizer
*Interactive portfolio optimization with efficient frontier visualization*

### Options Pricer
*Real-time options pricing with Greeks analysis*

### Market Data Explorer
*Historical price analysis and statistics*

## 🔮 Future Enhancements

- [ ] Backtesting framework with historical performance
- [ ] Value at Risk (VaR) and Conditional VaR calculations
- [ ] American options pricing using binomial tree model
- [ ] Multi-factor risk models (Fama-French)
- [ ] Machine learning for return prediction
- [ ] Real-time portfolio tracking and alerts
- [ ] PDF/CSV export functionality
- [ ] Authentication and multi-user support

## 🐛 Known Issues

- Yahoo Finance API may occasionally rate-limit requests (retry logic implemented)
- Optimization may use equal-weighted fallback for extreme parameters
- Historical data limited to publicly available tickers

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Modern Portfolio Theory** - Harry Markowitz (1952)
- **Black-Scholes-Merton Model** - Fischer Black, Myron Scholes, Robert Merton (1973)
- Market data provided by **Yahoo Finance**
- Built with **Streamlit** framework

## 👤 Author

**Ayotunde Akinboade**
- GitHub: [@lawren-ai](https://github.com/lawren-ai)
- LinkedIn: [Ayotunde Akinboade](https://linkedin.com/in/ayotunde-akinboade)


## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/lawren-ai/alpha-opt/issues).

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

---

*Built with ❤️ for the quantitative finance community*
```
