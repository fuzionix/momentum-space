import datetime
import pandas as pd
import yfinance as yf
from typing import List, Tuple, Optional

def fetch_stock_data(
    tickers: List[str], 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "5y"
) -> pd.DataFrame:
    """
    Fetch historical stock data for the given tickers.
    
    Args:
        tickers (List[str]): List of stock ticker symbols.
        start_date (Optional[str]): Start date in 'YYYY-MM-DD' format. If None, period is used.
        end_date (Optional[str]): End date in 'YYYY-MM-DD' format. If None, today is used.
        period (str): Period to fetch data for if start_date is None. Default is "5y" (5 years).
    
    Returns:
        output (pd.DataFrame): DataFrame with adjusted closing prices for each ticker.
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Fetch data based on dates or period
    if start_date:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    else:
        data = yf.download(tickers, period=period, end=end_date, progress=False)

    # Forward fill missing data and drop rows with all NaNs
    data = data.ffill().dropna(how="all")
    
    # Handle MultiIndex columns (multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        # Try to get 'Adj Close' for all tickers
        if 'Adj Close' in data.columns.levels[0]:
            adj_close = data['Adj Close']
            adj_close.columns = list(adj_close.columns)  # Ensure columns are ticker names
            return adj_close
        elif 'Close' in data.columns.levels[0]:
            close = data['Close']
            close.columns = list(close.columns)
            return close
        else:
            raise ValueError("找不到有效的價格欄位。")
    else:
        # Single ticker or flat columns
        if 'Adj Close' in data.columns:
            return pd.DataFrame(data['Adj Close'])
        elif 'Close' in data.columns:
            return pd.DataFrame(data['Close'])
        else:
            raise ValueError("找不到有效的價格欄位。")

def fetch_benchmark_data(
    benchmark_ticker: str = "^GSPC", # S&P 500 by default
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "5y"
) -> pd.Series:
    """
    Fetch benchmark data (default is S&P 500).
    
    Args:
        benchmark_ticker (str): Ticker symbol for the benchmark.
        start_date (Optional[str]): Start date in 'YYYY-MM-DD' format. If None, period is used.
        end_date (Optional[str]): End date in 'YYYY-MM-DD' format. If None, today is used.
        period (str): Period to fetch data for if start_date is None. Default is "5y" (5 years).
    
    Returns:
        output (pd.Series): Series with benchmark's adjusted closing prices.
    """
    benchmark_data = fetch_stock_data([benchmark_ticker], start_date, end_date, period)
    return benchmark_data[benchmark_ticker]


def validate_tickers(tickers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate if the provided tickers exist in Yahoo Finance.
    
    Args:
        tickers (List[str]): List of ticker symbols to validate.
    
    Returns:
        output (Tuple[List[str], List[str]]): Tuple of (valid_tickers, invalid_tickers).
    """
    valid, invalid = [], []
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="5d", progress=False)
            if data.empty:
                invalid.append(ticker)
            else:
                valid.append(ticker)
        except Exception:
            invalid.append(ticker)
    return valid, invalid