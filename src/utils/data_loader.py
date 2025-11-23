"""
Market Data Loader - Enhanced with retry logic and fallbacks
Fetches real stock prices and processes them for analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict
from datetime import datetime, timedelta
import time


class MarketDataLoader:
    """Load and process market data from Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
    
    def fetch_stock_data(self, tickers: List[str], 
                         start_date: str = None, 
                         end_date: str = None,
                         period: str = None,
                         max_retries: int = 3) -> pd.DataFrame:
        """
        Fetch historical stock prices with retry logic
        
        Parameters:
        -----------
        tickers : List[str]
            Stock symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
        start_date : str
            Start date 'YYYY-MM-DD' (default: 1 year ago)
        end_date : str
            End date 'YYYY-MM-DD' (default: today)
        max_retries : int
            Number of retry attempts
            
        Returns:
        --------
        pd.DataFrame : Adjusted close prices
        """
        if period and not start_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            years = int(period.replace("y", ""))  # e.g., "5y" → 5
            start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Ensure tickers is a list
        if isinstance(tickers, str):
            tickers = [tickers]
        
        all_prices = []
        
        for ticker in tickers:
            success = False
            
            for attempt in range(max_retries):
                try:
                    print(f"Fetching {ticker} (attempt {attempt + 1}/{max_retries})...")
                    
                    # Method 1: Use Ticker object (more reliable)
                    stock = yf.Ticker(ticker)
                    data = stock.history(start=start_date, end=end_date, auto_adjust=False)
                    
                    if data.empty:
                        print(f"  No data returned for {ticker}, trying alternative method...")
                        
                        # Method 2: Use download with different parameters
                        time.sleep(1)  # Add delay to avoid rate limiting
                        data = yf.download(
                            ticker,
                            start=start_date,
                            end=end_date,
                            progress=False,
                            auto_adjust=False,
                            timeout=10
                        )
                    
                    if data.empty:
                        if attempt < max_retries - 1:
                            print(f"  Empty data, retrying in 2 seconds...")
                            time.sleep(2)
                            continue
                        else:
                            print(f"Warning: No data available for {ticker}")
                            break
                    
                    # Extract price data
                    if 'Adj Close' in data.columns:
                        prices = data['Adj Close']
                    elif 'Close' in data.columns:
                        prices = data['Close']
                    else:
                        print(f"Warning: No price column found for {ticker}")
                        break
                    
                    # Convert to DataFrame
                    if isinstance(prices, pd.Series):
                        prices_df = prices.to_frame(name=ticker)
                    else:
                        prices_df = prices.copy()
                        if isinstance(prices_df, pd.DataFrame) and len(prices_df.columns) == 1:
                            prices_df.columns = [ticker]
                    
                    all_prices.append(prices_df)
                    print(f"  ✓ Successfully loaded {len(prices_df)} days of data for {ticker}")
                    success = True
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  Error on attempt {attempt + 1}: {e}")
                        print(f"  Retrying in {2 * (attempt + 1)} seconds...")
                        time.sleep(2 * (attempt + 1))
                    else:
                        print(f"Error downloading {ticker} after {max_retries} attempts: {e}")
            
            if success:
                time.sleep(0.5)  # Small delay between tickers to avoid rate limiting
        
        if not all_prices:
            raise ValueError(
                f"Could not download data for any ticker. "
                f"Possible causes:\n"
                f"1. Network/firewall blocking Yahoo Finance\n"
                f"2. Invalid ticker symbols\n"
                f"3. Yahoo Finance API rate limiting\n"
                f"4. No trading data in date range\n"
                f"Attempted tickers: {tickers}"
            )
        
        # Combine all DataFrames
        result = pd.concat(all_prices, axis=1)
        result = result.dropna(how='all')
        
        print(f"\n✓ Successfully loaded data for {len(result.columns)} ticker(s)")
        return result
    
    def calculate_returns(self, prices: pd.DataFrame, 
                          method: str = 'simple') -> pd.DataFrame:
        """Calculate returns from prices"""
        if method == 'simple':
            return prices.pct_change().dropna()
        elif method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
    
    def get_risk_free_rate(self) -> float:
        """Fetch current risk-free rate (10-year Treasury)"""
        try:
            treasury = yf.Ticker("^TNX")
            rate = treasury.history(period="1d")['Close'].iloc[-1] / 100
            return rate
        except:
            return 0.04  # Fallback to 4%
    
    def calculate_volatility(self, returns: pd.Series, 
                            window: int = 20) -> pd.Series:
        """Calculate rolling volatility (annualized)"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def fetch_options_data(self, ticker: str) -> Dict:
        """Fetch options chain data"""
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            
            if len(expirations) == 0:
                return None
            
            options = stock.option_chain(expirations[0])
            
            return {
                'calls': options.calls,
                'puts': options.puts,
                'expiration': expirations[0],
                'underlying_price': stock.history(period='1d')['Close'].iloc[-1]
            }
        except Exception as e:
            print(f"Error fetching options for {ticker}: {e}")
            return None


if __name__ == "__main__":
    loader = MarketDataLoader()
    
    print("=" * 70)
    print("TESTING MARKET DATA LOADER")
    print("=" * 70)
    
    # Test with a shorter date range (more reliable)
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    try:
        prices = loader.fetch_stock_data(
            test_tickers,
            start_date='2024-10-01',  # Recent date range
            end_date='2024-11-01'
        )
        
        print("\n" + "=" * 70)
        print("SUCCESS! Data loaded:")
        print("=" * 70)
        print(f"\nShape: {prices.shape}")
        print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        print(f"\nLast 5 days:\n{prices.tail()}")
        
        # Test returns calculation
        returns = loader.calculate_returns(prices)
        print(f"\nAnnualized Returns:\n{returns.mean() * 252}")
        print(f"\nAnnualized Volatility:\n{returns.std() * np.sqrt(252)}")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Try running: pip install --upgrade yfinance")
        print("3. Try using a VPN if Yahoo Finance is blocked")
        print("4. Check if you can access https://finance.yahoo.com in your browser")

