import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.executor import Executor


def test_size_order_with_margin(monkeypatch):
    # Simulate margins returned by Kite
    fake_margins = {'equity': {'available_cash': 10000}}

    def fake_fetch():
        return fake_margins

    monkeypatch.setattr('src.executor.fetch_margin_details', fake_fetch)
    e = Executor(simulated=True, cash=0)
    qty = e.size_order_with_margin('RELIANCE', price=100, side='buy', asset_class='equity', risk_per_trade=0.5)
    # available_cash=10000, allocation=5000 -> qty = 5000//100 = 50
    assert qty == 50
