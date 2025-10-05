from typing import List, Dict, Any, Optional
import pandas as pd


class Signal:
    def __init__(self, symbol: str, signal: str, confidence: float = 0.5, info: Optional[Dict[str, Any]] = None) -> None:
        self.symbol = symbol
        self.signal = signal  # 'buy', 'sell', 'hold'
        self.confidence = confidence
        self.info = info or {}

    def to_dict(self):
        return {"symbol": self.symbol, "signal": self.signal, "confidence": self.confidence, "info": self.info}


class Strategy:
    def generate(self, symbol: str, history: pd.DataFrame) -> Signal:
        raise NotImplementedError()


class MACrossoverStrategy(Strategy):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short = short_window
        self.long = long_window

    def generate(self, symbol: str, history: pd.DataFrame) -> Signal:
        if history is None or history.empty:
            return Signal(symbol, "hold", 0.0, {"reason": "no data"})
        close = history['close'].dropna()
        if len(close) < self.long:
            return Signal(symbol, "hold", 0.1, {"reason": "not enough data"})
        ma_short = close.rolling(self.short).mean().iloc[-1]
        ma_long = close.rolling(self.long).mean().iloc[-1]
        info = {"ma_short": float(ma_short), "ma_long": float(ma_long)}
        if ma_short > ma_long:
            return Signal(symbol, "buy", 0.6, info)
        elif ma_short < ma_long:
            return Signal(symbol, "sell", 0.6, info)
        else:
            return Signal(symbol, "hold", 0.5, info)


class RSIOLStrategy(Strategy):
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.period = period
        self.os = oversold
        self.ob = overbought

    def _rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(com=self.period - 1, adjust=False).mean()
        ma_down = down.ewm(com=self.period - 1, adjust=False).mean()
        rs = ma_up / ma_down
        return 100 - (100 / (1 + rs))

    def generate(self, symbol: str, history: pd.DataFrame) -> Signal:
        if history is None or history.empty:
            return Signal(symbol, "hold", 0.0, {"reason": "no data"})
        close = history['close'].dropna()
        if len(close) < self.period:
            return Signal(symbol, "hold", 0.1, {"reason": "not enough data"})
        rsi = self._rsi(close).iloc[-1]
        info = {"rsi": float(rsi)}
        if rsi < self.os:
            return Signal(symbol, "buy", 0.7, info)
        elif rsi > self.ob:
            return Signal(symbol, "sell", 0.7, info)
        else:
            return Signal(symbol, "hold", 0.5, info)
