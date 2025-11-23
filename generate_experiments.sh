#!/bin/bash

echo "Generating MLflow experiments..."

# Experiment 1: Conservative portfolio
curl -X 'POST' 'http://localhost:8000/api/v1/optimize-portfolio' \
  -H 'Content-Type: application/json' \
  -d '{
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "target_return": 0.08,
  "risk_tolerance": "conservative",
  "period": "5y"
}'
echo -e "\n✅ Experiment 1 complete\n"
sleep 1

# Experiment 2: Moderate portfolio
curl -X 'POST' 'http://localhost:8000/api/v1/optimize-portfolio' \
  -H 'Content-Type: application/json' \
  -d '{
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "target_return": 0.12,
  "risk_tolerance": "moderate",
  "period": "5y"
}'
echo -e "\n✅ Experiment 2 complete\n"
sleep 1

# Experiment 3: Aggressive portfolio
curl -X 'POST' 'http://localhost:8000/api/v1/optimize-portfolio' \
  -H 'Content-Type: application/json' \
  -d '{
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "target_return": 0.15,
  "risk_tolerance": "aggressive",
  "period": "5y"
}'
echo -e "\n✅ Experiment 3 complete\n"
sleep 1

# Experiment 4: Tech-heavy portfolio
curl -X 'POST' 'http://localhost:8000/api/v1/optimize-portfolio' \
  -H 'Content-Type: application/json' \
  -d '{
  "tickers": ["NVDA", "TSLA", "META", "GOOGL"],
  "target_return": 0.20,
  "risk_tolerance": "aggressive",
  "period": "3y"
}'
echo -e "\n✅ Experiment 4 complete\n"
sleep 1

# Experiment 5: Diversified portfolio
curl -X 'POST' 'http://localhost:8000/api/v1/optimize-portfolio' \
  -H 'Content-Type: application/json' \
  -d '{
  "tickers": ["AAPL", "JPM", "XOM", "JNJ", "WMT"],
  "target_return": 0.10,
  "risk_tolerance": "moderate",
  "period": "5y"
}'
echo -e "\n✅ Experiment 5 complete\n"
sleep 1

# Experiment 6: ATM Call Option
curl -X 'POST' 'http://localhost:8000/api/v1/price-option' \
  -H 'Content-Type: application/json' \
  -d '{
  "spot_price": 100,
  "strike_price": 100,
  "time_to_maturity": 1.0,
  "risk_free_rate": 0.05,
  "volatility": 0.20,
  "option_type": "call",
  "method": "black-scholes"
}'
echo -e "\n✅ Experiment 6 complete\n"
sleep 1

# Experiment 7: OTM Call Option
curl -X 'POST' 'http://localhost:8000/api/v1/price-option' \
  -H 'Content-Type: application/json' \
  -d '{
  "spot_price": 100,
  "strike_price": 110,
  "time_to_maturity": 1.0,
  "risk_free_rate": 0.05,
  "volatility": 0.20,
  "option_type": "call",
  "method": "black-scholes"
}'
echo -e "\n✅ Experiment 7 complete\n"
sleep 1

# Experiment 8: ITM Put Option
curl -X 'POST' 'http://localhost:8000/api/v1/price-option' \
  -H 'Content-Type: application/json' \
  -d '{
  "spot_price": 100,
  "strike_price": 110,
  "time_to_maturity": 0.5,
  "risk_free_rate": 0.05,
  "volatility": 0.30,
  "option_type": "put",
  "method": "black-scholes"
}'
echo -e "\n✅ Experiment 8 complete\n"
sleep 1

# Experiment 9: High volatility option
curl -X 'POST' 'http://localhost:8000/api/v1/price-option' \
  -H 'Content-Type: application/json' \
  -d '{
  "spot_price": 100,
  "strike_price": 100,
  "time_to_maturity": 1.0,
  "risk_free_rate": 0.05,
  "volatility": 0.50,
  "option_type": "call",
  "method": "black-scholes"
}'
echo -e "\n✅ Experiment 9 complete\n"
sleep 1

# Experiment 10: Short-dated option
curl -X 'POST' 'http://localhost:8000/api/v1/price-option' \
  -H 'Content-Type: application/json' \
  -d '{
  "spot_price": 100,
  "strike_price": 105,
  "time_to_maturity": 0.1,
  "risk_free_rate": 0.05,
  "volatility": 0.20,
  "option_type": "call",
  "method": "black-scholes"
}'
echo -e "\n✅ Experiment 10 complete\n"

echo "=========================================="
echo "✅ All 10 experiments complete!"
echo "=========================================="
echo "View results at: http://localhost:5000"