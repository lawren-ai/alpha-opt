

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import time
import mlflow
from datetime import datetime
import logging
import os

from src.portfolio.optimizer import PortfolioOptimizer
from src.options.black_scholes import BlackScholesModel
from src.options.monte_carlo import MonteCarloSimulator
from src.utils.data_loader import MarketDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("portfolio_optimization")
    logger.info(f"✅ MLflow configured: {MLFLOW_TRACKING_URI}")
except Exception as e:
    logger.warning(f"⚠️ MLflow setup failed: {e}")

app = FastAPI(
    title="AlphaOPT API",
    description="""
    Production-grade API for portfolio optimization, options pricing and strategy backtesting.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"{request.method} {request.url.path} - {process_time:.3f}s")
    return response

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# pydantic models 
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime_seconds: float

class PortfolioOptimizeRequest(BaseModel):
    """Request model for portfolio optimization"""
    tickers: List[str] = Field(..., 
        description="List of stock tickers (e.g., ['AAPL', 'MSFT', 'GOOGL'])",
        example=["AAPL", "MSFT", "GOOGL", "AMZN"]
    )
    target_return: Optional[float] = Field(None,
        description="Target annual return (e.g., 0.10 for 10%)",
        ge=0.0, le=1.0,
        example=0.12
    )
    risk_tolerance: Optional[str] = Field("moderate",
        description="Risk tolerance: 'conservative', 'moderate', or 'aggressive'",
        example="moderate"
    )
    period: Optional[str] = Field("5y",
        description="Historical data period (e.g., '1y', '5y', '10y')",
        example="5y"
    )
    max_weight: Optional[float] = Field(0.40,
        description="Maximum weight per asset (0.0 to 1.0)",
        ge=0.0, le=1.0,
        example=0.40
    )
    min_weight: Optional[float] = Field(0.05,
        description="Minimum weight per asset (0.0 to 1.0)",
        ge=0.0, le=1.0,
        example=0.05
    )
    
    @validator('tickers')
    def validate_tickers(cls, v):
        if len(v) < 2:
            raise ValueError("Must provide at least 2 tickers")
        if len(v) > 20:
            raise ValueError("Maximum 20 tickers allowed")
        return v
    
    @validator('risk_tolerance')
    def validate_risk_tolerance(cls, v):
        allowed = ['conservative', 'moderate', 'aggressive']
        if v not in allowed:
            raise ValueError(f"risk_tolerance must be one of {allowed}")
        return v

class PortfolioOptimizeResponse(BaseModel):
    """Response model for portfolio optimization"""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    concentration_hhi: float
    max_weight: float
    timestamp: str
    computation_time_ms: float

class OptionPriceRequest(BaseModel):
    """Request model for options pricing"""
    spot_price: float = Field(..., 
        description="Current stock price",
        gt=0,
        example=100.0
    )
    strike_price: float = Field(...,
        description="Option strike price",
        gt=0,
        example=105.0
    )
    time_to_maturity: float = Field(...,
        description="Time to maturity in years",
        gt=0,
        le=10,
        example=1.0
    )
    risk_free_rate: float = Field(...,
        description="Risk-free interest rate (annualized)",
        ge=0.0,
        le=0.50,
        example=0.05
    )
    volatility: float = Field(...,
        description="Implied volatility (annualized)",
        gt=0,
        le=5.0,
        example=0.20
    )
    option_type: str = Field(...,
        description="Option type: 'call' or 'put'",
        example="call"
    )
    method: Optional[str] = Field("black-scholes",
        description="Pricing method: 'black-scholes' or 'monte-carlo'",
        example="black-scholes"
    )
    mc_simulations: Optional[int] = Field(10000,
        description="Number of Monte Carlo simulations (if using MC method)",
        ge=1000,
        le=1000000,
        example=10000
    )
    
    @validator('option_type')
    def validate_option_type(cls, v):
        if v not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
        return v
    
    @validator('method')
    def validate_method(cls, v):
        if v not in ['black-scholes', 'monte-carlo']:
            raise ValueError("method must be 'black-scholes' or 'monte-carlo'")
        return v

class GreeksResponse(BaseModel):
    """Greeks response model"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class OptionPriceResponse(BaseModel):
    """Response model for options pricing"""
    option_price: float
    greeks: GreeksResponse
    method: str
    intrinsic_value: float
    time_value: float
    timestamp: str
    computation_time_ms: float
    # Monte Carlo specific fields
    mc_std_error: Optional[float] = None
    mc_confidence_interval: Optional[List[float]] = None

class BacktestRequest(BaseModel):
    """Request model for strategy backtesting"""
    tickers: List[str] = Field(...,
        description="List of stock tickers",
        example=["AAPL", "MSFT", "GOOGL"]
    )
    strategy: str = Field(...,
        description="Strategy type: 'equal-weight', 'max-sharpe', 'min-variance', 'risk-parity'",
        example="max-sharpe"
    )
    start_date: str = Field(...,
        description="Start date (YYYY-MM-DD)",
        example="2020-01-01"
    )
    end_date: str = Field(...,
        description="End date (YYYY-MM-DD)",
        example="2023-12-31"
    )
    rebalance_frequency: Optional[str] = Field("monthly",
        description="Rebalancing frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'",
        example="monthly"
    )
    initial_capital: Optional[float] = Field(10000.0,
        description="Initial capital in USD",
        gt=0,
        example=10000.0
    )
    
    @validator('strategy')
    def validate_strategy(cls, v):
        allowed = ['equal-weight', 'max-sharpe', 'min-variance', 'risk-parity']
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}")
        return v

class BacktestResponse(BaseModel):
    """Response model for backtesting"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    final_portfolio_value: float
    timestamp: str
    computation_time_ms: float

# api endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AlphaOPT API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "optimize_portfolio": "/api/v1/optimize-portfolio",
            "price_option": "/api/v1/price-option",
            "backtest_strategy": "/api/v1/backtest-strategy"
        }
    }

# startup time for health check
startup_time = time.time()

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint
    
    Returns API status, version, and uptime
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        uptime_seconds=time.time() - startup_time
    )

@app.post("/api/v1/optimize-portfolio", 
          response_model=PortfolioOptimizeResponse,
          tags=["Portfolio Optimization"])
async def optimize_portfolio(request: PortfolioOptimizeRequest):
    start_time = time.time()
    
    try:
        logger.info(f"Portfolio optimization request: {request.tickers}")
        
        # Load market data
        loader = MarketDataLoader()
        data = loader.fetch_stock_data(request.tickers, period=request.period)
        returns = loader.calculate_returns(data, method='log')

        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(returns)
        
        # Adjust constraints based on risk tolerance
        if request.risk_tolerance == 'conservative':
            max_weight = min(request.max_weight, 0.30)
            min_weight = max(request.min_weight, 0.10)
        elif request.risk_tolerance == 'moderate':
            max_weight = request.max_weight
            min_weight = request.min_weight
        else:  # aggressive
            max_weight = min(request.max_weight, 0.60)
            min_weight = 0.0
        
        # Optimize portfolio
        if request.target_return:
            result = optimizer.optimize(target_return=request.target_return)
        else:
            result = optimizer.max_sharpe_portfolio()
        
        # Calculate concentration metrics
        weights_array = np.array(list(result['weights'].values()))
        hhi = np.sum(weights_array**2)
        max_weight_actual = np.max(weights_array)
        
        computation_time = (time.time() - start_time) * 1000  # ms
        
        logger.info(f"Optimization completed in {computation_time:.2f}ms")
        # Log to MLflow
        try:
            with mlflow.start_run(run_name=f"portfolio_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("tickers", ",".join(request.tickers))
                mlflow.log_param("target_return", str(request.target_return))
                mlflow.log_param("risk_tolerance", request.risk_tolerance)
                mlflow.log_param("period", request.period)
                
                mlflow.log_metric("expected_return", float(result['return']))
                mlflow.log_metric("volatility", float(result['volatility']))
                mlflow.log_metric("sharpe_ratio", float(result['sharpe_ratio']))
                mlflow.log_metric("hhi", float(hhi))
                mlflow.log_metric("computation_time_ms", computation_time)
                
                mlflow.log_dict(result['weights'], "weights.json")
                
                mlflow.set_tag("model_type", "mean_variance")
                
                logger.info("✅ Logged to MLflow")
        except Exception as mlflow_error:
            logger.warning(f"⚠️ MLflow logging failed: {mlflow_error}")
        return PortfolioOptimizeResponse(
            weights=result['weights'],
            expected_return=result['return'],
            volatility=result['volatility'],
            sharpe_ratio=result['sharpe_ratio'],
            concentration_hhi=hhi,
            max_weight=max_weight_actual,
            timestamp=datetime.utcnow().isoformat(),
            computation_time_ms=computation_time
        )
        
    except Exception as e:
        logger.error(f"Portfolio optimization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/api/v1/price-option",
          response_model=OptionPriceResponse,
          tags=["Options Pricing"])
async def price_option(request: OptionPriceRequest):
    start_time = time.time()
    
    try:
        logger.info(f"Option pricing request: {request.option_type} @ {request.strike_price}")
        
        # Calculate intrinsic value
        if request.option_type == 'call':
            intrinsic = max(request.spot_price - request.strike_price, 0)
        else:
            intrinsic = max(request.strike_price - request.spot_price, 0)
        
        if request.method == 'black-scholes':
            bs = BlackScholesModel(
                S=request.spot_price,
                K=request.strike_price,
                T=request.time_to_maturity,
                r=request.risk_free_rate,
                sigma=request.volatility
            )
            
            greeks = bs.all_greeks(request.option_type)
            option_price = greeks['price']
            
            computation_time = (time.time() - start_time) * 1000
            
            logger.info(f"Black-Scholes pricing completed in {computation_time:.2f}ms")
                        # Log to MLflow  
            try:
                with mlflow.start_run(run_name=f"option_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_param("spot_price", request.spot_price)
                    mlflow.log_param("strike_price", request.strike_price)
                    mlflow.log_param("option_type", request.option_type)
                    mlflow.log_param("method", "black-scholes")
                    
                    mlflow.log_metric("option_price", float(option_price))
                    mlflow.log_metric("delta", float(greeks['delta']))
                    
                    mlflow.set_tag("model_type", "black_scholes")
                    
                    logger.info("✅ Logged to MLflow")
            except Exception as e:
                logger.warning(f"⚠️ MLflow logging failed: {e}")

            return OptionPriceResponse(
                option_price=option_price,
                greeks=GreeksResponse(
                    delta=greeks['delta'],
                    gamma=greeks['gamma'],
                    theta=greeks['theta'],
                    vega=greeks['vega'],
                    rho=greeks['rho']
                ),
                method="black-scholes",
                intrinsic_value=intrinsic,
                time_value=option_price - intrinsic,
                timestamp=datetime.utcnow().isoformat(),
                computation_time_ms=computation_time
            )
            
        else:  
            mc = MonteCarloSimulator(
                S0=request.spot_price,
                K=request.strike_price,
                T=request.time_to_maturity,
                r=request.risk_free_rate,
                sigma=request.volatility,
                n_simulations=request.mc_simulations
            )
            
            mc_result = mc.price_european_option(request.option_type)
            option_price = mc_result['price']
            
            # Calculate Greeks using finite differences
            greeks_mc = mc.calculate_greeks_mc(request.option_type)
            
            # get Black-Scholes Greeks for comparison
            bs = BlackScholesModel(
                S=request.spot_price,
                K=request.strike_price,
                T=request.time_to_maturity,
                r=request.risk_free_rate,
                sigma=request.volatility
            )
            
            bs_theta = bs.theta(request.option_type)
            bs_rho = bs.rho(request.option_type)
            
            computation_time = (time.time() - start_time) * 1000
            
            logger.info(f"Monte Carlo pricing completed in {computation_time:.2f}ms")
            
            return OptionPriceResponse(
                option_price=option_price,
                greeks=GreeksResponse(
                    delta=greeks_mc['delta'],
                    gamma=greeks_mc['gamma'],
                    theta=bs_theta,  # Use BS for theta (more accurate)
                    vega=greeks_mc['vega'],
                    rho=bs_rho  # Use BS for rho (more accurate)
                ),
                method="monte-carlo",
                intrinsic_value=intrinsic,
                time_value=option_price - intrinsic,
                timestamp=datetime.utcnow().isoformat(),
                computation_time_ms=computation_time,
                mc_std_error=mc_result['std_error'],
                mc_confidence_interval=[mc_result['conf_lower'], mc_result['conf_upper']]
            )
        
    except Exception as e:
        logger.error(f"Option pricing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pricing failed: {str(e)}")

@app.post("/api/v1/backtest-strategy",
          response_model=BacktestResponse,
          tags=["Backtesting"])
async def backtest_strategy(request: BacktestRequest):
    start_time = time.time()
    
    try:
        logger.info(f"Backtest request: {request.strategy} on {request.tickers}")
        
        # Load data
        loader = MarketDataLoader()
        data = loader.fetch_stock_data(
            request.tickers,
            start_date=request.start_date,
            end_date=request.end_date
        )
        returns = loader.calculate_returns(data, method='log')
        
        portfolio_value = request.initial_capital
        portfolio_values = [portfolio_value]
        
        # Equal weight strategy (simplified)
        n_assets = len(request.tickers)
        daily_returns = returns.mean(axis=1)  # Portfolio return
        
        for daily_return in daily_returns:
            portfolio_value *= (1 + daily_return)
            portfolio_values.append(portfolio_value)
        
        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        total_return = (portfolio_values[-1] - request.initial_capital) / request.initial_capital
        
        # Annualized return
        n_days = len(portfolio_values) - 1
        years = n_days / 252
        annualized_return = (portfolio_values[-1] / request.initial_capital) ** (1/years) - 1
        
        # Volatility
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cummax) / cummax
        max_drawdown = abs(np.min(drawdowns))
        
        # Win rate
        wins = np.sum(portfolio_returns > 0)
        win_rate = wins / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
        
        # Number of trades (simplified: assume rebalancing at specified frequency)
        num_trades = int(n_days / {'daily': 1, 'weekly': 5, 'monthly': 21, 'quarterly': 63, 'yearly': 252}[request.rebalance_frequency])
        
        computation_time = (time.time() - start_time) * 1000
        
        logger.info(f"Backtest completed in {computation_time:.2f}ms")
        
        return BacktestResponse(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_trades=num_trades,
            final_portfolio_value=portfolio_values[-1],
            timestamp=datetime.utcnow().isoformat(),
            computation_time_ms=computation_time
        )
        
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

# startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 60)
    logger.info("AlphaOPT API Starting Up")
    logger.info("=" * 60)
    logger.info(f"FastAPI version: {app.version}")
    logger.info(f"Documentation: http://localhost:8000/docs")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("AlphaOPT API Shutting Down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )