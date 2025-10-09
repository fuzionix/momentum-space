import pandas as pd
import yfinance as yf
from typing import List, Tuple
from cachetools import TTLCache, cached

CACHE_TTL = 43200  # 12 hours in seconds
ticker_cache = TTLCache(maxsize=256, ttl=CACHE_TTL)

def fetch_stock_data(
    tickers: List[str],
    period: str = "5y",
) -> pd.DataFrame:
    all_data = []
    for ticker in tickers:
        try:
            df = fetch_single_ticker_cached(ticker, period)
            all_data.append(df)
        except Exception as e:
            raise ValueError(f"無法獲取 {ticker} 的價格資料。")

    if not all_data:
        raise ValueError("無法取得任何有效股票資料。")

    data = pd.concat(all_data, axis=1)

    data = data.ffill().dropna(how="all")
    data = data.sort_index()
    data.index = pd.to_datetime(data.index).tz_localize(None)  # Normalize timezone

    return data

def fetch_benchmark_data(
    benchmark_ticker: str = "^GSPC",
    period: str = "5y",
) -> pd.Series:
    df = fetch_stock_data([benchmark_ticker], period)
    return df[benchmark_ticker]

def validate_tickers(tickers: List[str]) -> Tuple[List[str], List[str]]:
    valid, invalid = [], []
    for ticker in tickers:
        try:
            df = fetch_single_ticker_cached(ticker, "5d")
            if df is not None and not df.empty:
                valid.append(ticker)
            else:
                invalid.append(ticker)
        except Exception:
            invalid.append(ticker)
    return valid, invalid

def cache_key(ticker, period):
    return f"{ticker}:{period}"

def ticker_cache_key(ticker, period):
    return cache_key(ticker, period)

@cached(cache=ticker_cache, key=ticker_cache_key)
def fetch_single_ticker_cached(ticker: str, period: str = "5y") -> pd.DataFrame:
    # Fetch data for a single ticker
    data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    
    if data is None or data.empty:
        raise ValueError(f"無法獲取 {ticker} 的價格資料。")

    # Handle MultiIndex columns (multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        # Try to get 'Adj Close' for all tickers
        if "Adj Close" in data.columns.levels[0]:
            adj_close = data["Adj Close"]
        elif "Close" in data.columns.levels[0]:
            adj_close = data["Close"]
        else:
            raise ValueError(f"缺少有效的收盤價格欄位。")
        adj_close.columns = [ticker] if isinstance(adj_close, pd.Series) else [ticker]
        return adj_close
    else:
        # Single ticker or flat columns
        if "Adj Close" in data.columns:
            series = data["Adj Close"]
        elif "Close" in data.columns:
            series = data["Close"]
        else:
            raise ValueError(f"缺少有效的收盤價格欄位。")

    df = series.to_frame(name=ticker)
    return df

def clear_cache():
    ticker_cache.clear()