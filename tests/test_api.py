

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import app

client = TestClient(app)

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data
    assert "uptime_seconds" in data

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AlphaOPT API"
    assert data["status"] == "operational"

def test_optimize_portfolio():
    """Test portfolio optimization endpoint"""
    payload = {
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "risk_tolerance": "moderate",
        "period": "1y"
    }
    
    response = client.post("/api/v1/optimize-portfolio", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "weights" in data
    assert "sharpe_ratio" in data
    assert "expected_return" in data
    assert "volatility" in data
    assert "computation_time_ms" in data
    
    # Check weights sum to approximately 1
    weights = list(data['weights'].values())
    assert abs(sum(weights) - 1.0) < 0.01

def test_price_option_black_scholes():
    """Test Black-Scholes option pricing"""
    payload = {
        "spot_price": 100.0,
        "strike_price": 105.0,
        "time_to_maturity": 1.0,
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "option_type": "call",
        "method": "black-scholes"
    }
    
    response = client.post("/api/v1/price-option", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.json()}")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "option_price" in data
    assert "greeks" in data
    assert data["greeks"]["delta"] > 0  # Call delta should be positive
    assert data["method"] == "black-scholes"
    
    # Check computation time is reasonable
    assert data["computation_time_ms"] < 100  # BS should be very fast

def test_price_option_monte_carlo():
    """Test Monte Carlo option pricing"""
    payload = {
        "spot_price": 100.0,
        "strike_price": 100.0,
        "time_to_maturity": 1.0,
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "option_type": "call",
        "method": "monte-carlo",
        "mc_simulations": 10000
    }
    
    response = client.post("/api/v1/price-option", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.json()}")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "option_price" in data
    assert "mc_std_error" in data
    assert "mc_confidence_interval" in data
    assert len(data["mc_confidence_interval"]) == 2

def test_backtest_strategy():
    """Test strategy backtesting"""
    payload = {
        "tickers": ["AAPL", "MSFT"],
        "strategy": "max-sharpe",
        "start_date": "2023-01-01",
        "end_date": "2023-06-01",
        "rebalance_frequency": "monthly",
        "initial_capital": 10000.0
    }
    
    response = client.post("/api/v1/backtest-strategy", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.json()}")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "total_return" in data
    assert "sharpe_ratio" in data
    assert "max_drawdown" in data
    assert "final_portfolio_value" in data

def test_invalid_tickers():
    """Test error handling for invalid tickers"""
    payload = {
        "tickers": ["A"],  # Only 1 ticker (minimum is 2)
        "period": "1y"
    }
    
    response = client.post("/api/v1/optimize-portfolio", json=payload)
    assert response.status_code == 422  # Validation error

def test_invalid_option_type():
    """Test error handling for invalid option type"""
    payload = {
        "spot_price": 100.0,
        "strike_price": 105.0,
        "time_to_maturity": 1.0,
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "option_type": "invalid",  # Invalid type
        "method": "black-scholes"
    }
    
    response = client.post("/api/v1/price-option", json=payload)
    assert response.status_code == 422  # Validation error

def test_api_documentation():
    """Test that API documentation is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi_schema():
    """Test OpenAPI schema endpoint"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])