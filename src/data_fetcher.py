from typing import Optional
import pandas as pd
import datetime as dt
import os
import logging

logger = logging.getLogger('atwz.data_fetcher')

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from nsepy import get_history
except Exception:
    get_history = None


def _cache_path(symbol: str, period: str, interval: str) -> str:
    safe_symbol = symbol.replace('/', '_')
    return os.path.join('.cache', f"{safe_symbol}.{period}.{interval}.csv")


def fetch_history(symbol: str, period: str = "1y", interval: str = "1d", use_cache: bool = True) -> pd.DataFrame:
    """Fetch historical OHLCV for a symbol.

    Attempts yfinance first, then nsepy for NSE tickers (like 'RELIANCE').
    """
    os.makedirs('.cache', exist_ok=True)
    cache = _cache_path(symbol, period, interval)
    if use_cache and os.path.exists(cache):
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            return df
        except Exception:
            logger.debug("Failed to read cache %s", cache, exc_info=True)

    df = pd.DataFrame()
    if yf is not None:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                })
        except Exception:
            logger.exception("yfinance fetch failed for %s", symbol)
            df = pd.DataFrame()

    # if empty and nsepy available and symbol looks like NSE (no dot)
    if (df is None or df.empty) and get_history is not None and '.' not in symbol:
        try:
            end = dt.date.today()
            if period.endswith('y'):
                years = int(period[:-1])
                start = dt.date(end.year - years, end.month, end.day)
            else:
                start = end - dt.timedelta(days=365)
            raw = get_history(symbol=symbol, start=start, end=end)
            if not raw.empty:
                raw = raw.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                })
                df = raw
        except Exception:
            logger.exception("nsepy fetch failed for %s", symbol)
            df = pd.DataFrame()

    if not df.empty:
        try:
            df.to_csv(cache)
        except Exception:
            logger.debug("Failed to write cache %s", cache, exc_info=True)

    return df
