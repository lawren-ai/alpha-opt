"""
Quantitative Finance Toolkit - Streamlit Dashboard
Interactive portfolio optimization and options pricing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.append('src')

from portfolio.optimizer import PortfolioOptimizer
from options.black_scholes import BlackScholesModel
from options.monte_carlo import MonteCarloSimulator
from utils.data_loader import MarketDataLoader

# Page configuration
st.set_page_config(
    page_title="AlphaOPT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'prices' not in st.session_state:
    st.session_state.prices = None
if 'returns' not in st.session_state:
    st.session_state.returns = None

# Title
st.title(" AlphaOPT")
st.markdown("*Portfolio Optimization & Options Pricing Platform*")
st.markdown("---")

# Sidebar - Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module:",
    [" Portfolio Optimizer", " Options Pricer", " Market Data Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This toolkit implements:\n"
    "- Mean-Variance Portfolio Optimization\n"
    "- Black-Scholes Option Pricing\n"
    "- Monte Carlo Simulations\n"
    "- Real-time Market Data Analysis"
)

# ============================================================================
# PAGE 1: PORTFOLIO OPTIMIZER
# ============================================================================
if page == " Portfolio Optimizer":
    st.header("Portfolio Optimization")
    st.markdown("Optimize your portfolio using Modern Portfolio Theory")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Input Parameters")
        
        # Ticker input
        default_tickers = "AAPL,MSFT,GOOGL,AMZN"
        tickers_input = st.text_input(
            "Stock Tickers (comma-separated)",
            value=default_tickers,
            help="Enter stock symbols separated by commas"
        )
        tickers = [t.strip().upper() for t in tickers_input.split(',')]
        
        # Date range
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=730),
                max_value=datetime.now()
            )
        with col_date2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # Load data button
        if st.button(" Load Data", type="primary"):
            with st.spinner("Fetching market data..."):
                try:
                    loader = MarketDataLoader()
                    prices = loader.fetch_stock_data(
                        tickers,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                    returns = loader.calculate_returns(prices)
                    
                    st.session_state.prices = prices
                    st.session_state.returns = returns
                    st.session_state.data_loaded = True
                    st.success(f" Loaded {len(prices)} days of data for {len(tickers)} stocks")
                except Exception as e:
                    st.error(f" Error loading data: {e}")
    
    with col2:
        st.subheader(" Optimization Settings")
        target_return = st.slider(
            "Target Annual Return (%)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=1.0
        ) / 100
        
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1
        ) / 100
    
    # Show data if loaded
    if st.session_state.data_loaded:
        st.markdown("---")
        
        # Display price chart
        st.subheader(" Price History")
        fig = go.Figure()
        for col in st.session_state.prices.columns:
            # Normalize to 100 for comparison
            normalized = (st.session_state.prices[col] / st.session_state.prices[col].iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=st.session_state.prices.index,
                y=normalized,
                mode='lines',
                name=col
            ))
        fig.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Run optimization
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button(" Optimize Portfolio", type="primary", use_container_width=True):
                with st.spinner("Running optimization..."):
                    try:
                        optimizer = PortfolioOptimizer(st.session_state.returns)
                        result = optimizer.optimize(
                            target_return=target_return,
                            risk_free_rate=risk_free_rate
                        )
                        
                        # Display results
                        st.success(" Optimization Complete!")
                        
                        # Metrics
                        st.markdown("###  Portfolio Metrics")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                "Expected Return",
                                f"{result['return']*100:.2f}%",
                                delta=f"{(result['return']-risk_free_rate)*100:.2f}% vs Risk-Free"
                            )
                        
                        with metric_col2:
                            st.metric(
                                "Volatility (Risk)",
                                f"{result['volatility']*100:.2f}%"
                            )
                        
                        with metric_col3:
                            st.metric(
                                "Sharpe Ratio",
                                f"{result['sharpe_ratio']:.3f}",
                                help="Higher is better. >1 is good, >2 is excellent"
                            )
                        
                        # Weights visualization
                        st.markdown("###  Optimal Portfolio Allocation")
                        
                        col_pie, col_table = st.columns([1, 1])
                        
                        with col_pie:
                            weights_df = pd.DataFrame({
                                'Asset': list(result['weights'].keys()),
                                'Weight': [w*100 for w in result['weights'].values()]
                            })
                            
                            fig_pie = px.pie(
                                weights_df,
                                values='Weight',
                                names='Asset',
                                title='Portfolio Weights (%)'
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col_table:
                            st.markdown("#### Allocation Table")
                            weights_display = pd.DataFrame({
                                'Asset': list(result['weights'].keys()),
                                'Weight (%)': [f"{w*100:.2f}%" for w in result['weights'].values()],
                                'Amount ($)': [f"${w*10000:.2f}" for w in result['weights'].values()]
                            })
                            st.dataframe(weights_display, use_container_width=True, hide_index=True)
                            st.caption("*Based on $10,000 portfolio")
                        
                        # Efficient Frontier
                        st.markdown("###  Efficient Frontier")
                        with st.spinner("Generating efficient frontier..."):
                            frontier = optimizer.efficient_frontier(n_points=50)
                            
                            fig_frontier = go.Figure()
                            
                            # Efficient frontier line
                            fig_frontier.add_trace(go.Scatter(
                                x=frontier['volatility']*100,
                                y=frontier['return']*100,
                                mode='lines',
                                name='Efficient Frontier',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Optimal portfolio point
                            fig_frontier.add_trace(go.Scatter(
                                x=[result['volatility']*100],
                                y=[result['return']*100],
                                mode='markers',
                                name='Optimal Portfolio',
                                marker=dict(color='red', size=15, symbol='star')
                            ))
                            
                            fig_frontier.update_layout(
                                title='Efficient Frontier - Risk vs Return',
                                xaxis_title='Volatility (Risk) %',
                                yaxis_title='Expected Return %',
                                hovermode='closest',
                                height=500
                            )
                            
                            st.plotly_chart(fig_frontier, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f" Optimization failed: {e}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())

# ============================================================================
# PAGE 2: OPTIONS PRICER
# ============================================================================
elif page == " Options Pricer":
    st.header("Options Pricing & Greeks")
    st.markdown("Price options using Black-Scholes Model and Monte Carlo Simulation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(" Option Parameters")
        
        # Stock ticker for reference
        ticker = st.text_input("Stock Ticker (optional)", value="AAPL", help="For reference only")
        
        S = st.number_input(
            "Current Stock Price ($)",
            min_value=0.01,
            value=150.0,
            step=1.0,
            help="Spot price of the underlying asset"
        )
        
        K = st.number_input(
            "Strike Price ($)",
            min_value=0.01,
            value=150.0,
            step=1.0,
            help="Exercise price of the option"
        )
        
        T = st.slider(
            "Time to Expiration (years)",
            min_value=0.01,
            max_value=5.0,
            value=0.25,
            step=0.01,
            help="Time remaining until option expiration"
        )
        
        r = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1
        ) / 100
        
        sigma = st.slider(
            "Volatility (%)",
            min_value=1.0,
            max_value=200.0,
            value=30.0,
            step=1.0,
            help="Annualized volatility (standard deviation of returns)"
        ) / 100
        
        option_type = st.radio("Option Type", ["Call", "Put"])
    
    with col2:
        st.subheader(" Pricing Method")
        method = st.radio(
            "Select Method:",
            ["Black-Scholes (Analytical)", "Monte Carlo (Simulation)"]
        )
        
        if method == "Monte Carlo (Simulation)":
            n_simulations = st.select_slider(
                "Number of Simulations",
                options=[10000, 50000, 100000, 250000, 500000],
                value=100000
            )
        
        if st.button(" Price Option", type="primary", use_container_width=True):
            with st.spinner("Calculating..."):
                try:
                    if method == "Black-Scholes (Analytical)":
                        bs = BlackScholesModel(S, K, T, r, sigma)
                        
                        if option_type == "Call":
                            price = bs.call_price()
                            greeks = bs.all_greeks('call')
                        else:
                            price = bs.put_price()
                            greeks = bs.all_greeks('put')
                        
                        st.success(" Calculation Complete!")
                        
                        # Display price
                        st.markdown("###  Option Price")
                        st.metric(
                            f"{option_type} Option Price",
                            f"${price:.4f}",
                            help="Theoretical fair value"
                        )
                        
                        # Display Greeks
                        st.markdown("###  The Greeks")
                        st.markdown("*Sensitivity measures for risk management*")
                        
                        greek_col1, greek_col2, greek_col3 = st.columns(3)
                        
                        with greek_col1:
                            st.metric(
                                "Delta (Δ)",
                                f"{greeks['delta']:.4f}",
                                help="Change in option price per $1 change in stock price"
                            )
                            st.metric(
                                "Gamma (Γ)",
                                f"{greeks['gamma']:.4f}",
                                help="Rate of change of Delta"
                            )
                        
                        with greek_col2:
                            st.metric(
                                "Vega (ν)",
                                f"{greeks['vega']:.4f}",
                                help="Change in option price per 1% change in volatility"
                            )
                            st.metric(
                                "Theta (Θ)",
                                f"{greeks['theta']:.4f}",
                                help="Daily time decay (usually negative)"
                            )
                        
                        with greek_col3:
                            st.metric(
                                "Rho (ρ)",
                                f"{greeks['rho']:.4f}",
                                help="Change in option price per 1% change in interest rate"
                            )
                        
                        # Greeks explanation
                        with st.expander(" Understanding the Greeks"):
                            st.markdown("""
                            **Delta**: How much the option price changes when stock price moves $1
                            - Call: 0 to 1 | Put: -1 to 0
                            - At-the-money options have Delta near ±0.5
                            
                            **Gamma**: How much Delta changes when stock price moves $1
                            - Highest for at-the-money options near expiration
                            - Measures convexity/curvature
                            
                            **Vega**: How much option price changes per 1% volatility change
                            - Always positive for long options
                            - Highest for at-the-money options
                            
                            **Theta**: Daily time decay
                            - Usually negative (options lose value over time)
                            - Accelerates as expiration approaches
                            
                            **Rho**: How much option price changes per 1% interest rate change
                            - More significant for longer-dated options
                            """)
                    
                    else:  # Monte Carlo
                        mc = MonteCarloSimulator(S, K, T, r, sigma, n_simulations=n_simulations)
                        
                        if option_type == "Call":
                            result = mc.price_european_call()
                        else:
                            result = mc.price_european_put()
                        
                        st.success(" Simulation Complete!")
                        
                        # Display results
                        st.markdown("###  Option Price (Monte Carlo)")
                        
                        price_col1, price_col2 = st.columns(2)
                        
                        with price_col1:
                            st.metric(
                                f"{option_type} Option Price",
                                f"${result['price']:.4f}"
                            )
                        
                        with price_col2:
                            st.metric(
                                "95% Confidence Interval",
                                f"±${result['confidence_interval']:.4f}",
                                help="Range of uncertainty in the estimate"
                            )
                        
                        st.info(f" Based on {n_simulations:,} simulated price paths")
                        
                        # Compare with Black-Scholes
                        bs = BlackScholesModel(S, K, T, r, sigma)
                        bs_price = bs.call_price() if option_type == "Call" else bs.put_price()
                        
                        st.markdown("###  Comparison with Black-Scholes")
                        comp_col1, comp_col2, comp_col3 = st.columns(3)
                        
                        with comp_col1:
                            st.metric("Monte Carlo", f"${result['price']:.4f}")
                        with comp_col2:
                            st.metric("Black-Scholes", f"${bs_price:.4f}")
                        with comp_col3:
                            diff = abs(result['price'] - bs_price)
                            st.metric("Difference", f"${diff:.4f}")
                
                except Exception as e:
                    st.error(f" Calculation failed: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

# ============================================================================
# PAGE 3: MARKET DATA EXPLORER
# ============================================================================
else:  # Market Data Explorer
    st.header("Market Data Explorer")
    st.markdown("Analyze historical price data and statistics")
    
    # Input
    ticker = st.text_input("Stock Ticker", value="AAPL")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
    
    if st.button(" Load Data", type="primary"):
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                loader = MarketDataLoader()
                prices = loader.fetch_stock_data(
                    [ticker],
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                returns = loader.calculate_returns(prices)
                
                st.success(f" Loaded {len(prices)} days of data")
                
                # Price chart
                st.markdown("###  Price History")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=prices.index,
                    y=prices[ticker],
                    mode='lines',
                    name='Price',
                    fill='tozeroy'
                ))
                fig.update_layout(
                    title=f"{ticker} Price History",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.markdown("###  Statistics")
                
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.metric("Current Price", f"${prices[ticker].iloc[-1]:.2f}")
                with stat_col2:
                    daily_return = returns[ticker].mean() * 252
                    st.metric("Annual Return", f"{daily_return*100:.2f}%")
                with stat_col3:
                    volatility = returns[ticker].std() * np.sqrt(252)
                    st.metric("Volatility", f"{volatility*100:.2f}%")
                with stat_col4:
                    max_price = prices[ticker].max()
                    min_price = prices[ticker].min()
                    st.metric("High", f"${max_price:.2f}")
                
                # Returns distribution
                st.markdown("###  Returns Distribution")
                fig_hist = px.histogram(
                    returns,
                    x=ticker,
                    nbins=50,
                    title="Daily Returns Distribution"
                )
                fig_hist.update_layout(
                    xaxis_title="Daily Return",
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
            except Exception as e:
                st.error(f" Error: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built with Streamlit • Data from Yahoo Finance • "
    "Models: Black-Scholes, Mean-Variance Optimization, Monte Carlo"
    "</div>",
    unsafe_allow_html=True
)