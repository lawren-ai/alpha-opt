# Quantitative Finance Toolkit

A comprehensive Python-based toolkit for portfolio optimization and derivatives pricing, featuring real-time market data integration and interactive visualizations.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

### Portfolio Optimization
- **Mean-Variance Optimization** (Markowitz, 1952)
- Efficient frontier visualization
- Risk-return trade-off analysis
- Sharpe ratio maximization
- Customizable constraints (position limits, target returns)

### Options Pricing
- **Black-Scholes Model** for European options
- Complete Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- **Monte Carlo simulation** for exotic options
- Pricing comparison between analytical and numerical methods

### Market Data Integration
- Real-time data from Yahoo Finance
- Historical price analysis
- Returns and volatility calculations
- Automatic retry logic for reliability

### Interactive Dashboard
- Built with Streamlit
- Real-time parameter adjustment
- Interactive visualizations (Plotly)
- User-friendly interface

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pip
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/quant-finance-toolkit.git
cd quant-finance-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## 📊 Usage Examples

### Portfolio Optimization
```python
from src.portfolio.optimizer import PortfolioOptimizer
from src.utils.data_loader import MarketDataLoader

# Load market data
loader = MarketDataLoader()
prices = loader.fetch_stock_data(['AAPL', 'MSFT', 'GOOGL'], 
                                 start_date='2023-01-01')
returns = loader.calculate_returns(prices)

# Optimize portfolio
optimizer = PortfolioOptimizer(returns)
result = optimizer.optimize(target_return=0.15)

print(f"Optimal Weights: {result['weights']}")
print(f"Expected Return: {result['return']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
```

### Options Pricing
```python
from src.options.black_scholes import BlackScholesModel

# Price a call option
bs = BlackScholesModel(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
call_price = bs.call_price()
greeks = bs.all_greeks('call')

print(f"Call Price: ${call_price:.2f}")
print(f"Delta: {greeks['delta']:.4f}")
```

## 🛠️ Tech Stack

- **Python 3.9+** - Core language
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **CVXPY** - Convex optimization
- **SciPy** - Scientific computing
- **Streamlit** - Web dashboard
- **Plotly** - Interactive visualizations
- **yfinance** - Market data

## 📁 Project Structure
```
quant-finance-toolkit/
├── src/
│   ├── portfolio/
│   │   ├── optimizer.py          # Mean-variance optimization
│   │   └── efficient_frontier.py
│   ├── options/
│   │   ├── black_scholes.py      # BS pricing model
│   │   ├── greeks.py             # Greeks calculator
│   │   └── monte_carlo.py        # MC simulation
│   └── utils/
│       └── data_loader.py        # Market data fetching
├── tests/
│   └── test_integration.py
├── app.py                        # Streamlit dashboard
├── requirements.txt
└── README.md
```

## 🎓 Theory & Concepts

### Modern Portfolio Theory
Implements Harry Markowitz's mean-variance optimization framework:
- Minimizes portfolio risk for a given return
- Accounts for asset correlations
- Generates efficient frontier

### Black-Scholes Model
Closed-form solution for European options pricing:
- Based on geometric Brownian motion
- Calculates all Greeks analytically
- Industry standard for option valuation

### Monte Carlo Simulation
Numerical method for complex derivatives:
- Simulates thousands of price paths
- Can price path-dependent options
- Flexible for custom payoff structures

## 🔮 Future Enhancements

- [ ] Backtesting framework
- [ ] Value at Risk (VaR) calculations
- [ ] American options pricing (binomial model)
- [ ] Machine learning for return prediction
- [ ] Real-time portfolio tracking
- [ ] Export to PDF/CSV

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

Your Name - [LinkedIn](https://linkedin.com/in/yourpr) | [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Based on Modern Portfolio Theory (Markowitz, 1952)
- Black-Scholes-Merton model (Black & Scholes, 1973)
- Data provided by Yahoo Finance