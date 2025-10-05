import os
import sys
import pandas as pd
# ensure src is importable when tests are run from workspace root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from strategy import MACrossoverStrategy


def test_macrossover_basic():
    idx = pd.date_range("2020-01-01", periods=100)
    prices = pd.Series([i for i in range(100)], index=idx)
    df = pd.DataFrame({'close': prices, 'open': prices, 'high': prices, 'low': prices, 'volume': 0})
    strat = MACrossoverStrategy(short_window=5, long_window=20)
    sig = strat.generate("FAKE", df)
    assert sig.signal in ("buy", "sell", "hold")
