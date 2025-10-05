from typing import List, Dict
import pandas as pd
from dataclasses import dataclass


@dataclass
class Trade:
    symbol: str
    entry_price: float
    exit_price: float | None = None
    qty: int = 0
    profit: float = 0.0


class SimpleBacktester:
    def __init__(self, starting_cash: float = 100000.0):
        self.cash: float = starting_cash
        self.positions: Dict[str, Trade] = {}
        self.trades: List[Trade] = []

    def run(self, symbol: str, history: pd.DataFrame, signals: List[Dict]):
        # signals: list of {index, signal, qty}
        for s in signals:
            idx = s.get('index')
            sig = s.get('signal')
            qty = s.get('qty', 0)
            price = float(history['close'].iloc[idx])
            if sig == 'buy':
                if symbol in self.positions:
                    continue
                cost = price * qty
                if cost > self.cash:
                    continue
                tr = Trade(symbol=symbol, entry_price=price, qty=qty)
                self.positions[symbol] = tr
                self.cash -= cost
            elif sig == 'sell':
                if symbol not in self.positions:
                    continue
                tr = self.positions.pop(symbol)
                tr.exit_price = price
                tr.profit = (tr.exit_price - tr.entry_price) * tr.qty
                self.cash += price * tr.qty
                self.trades.append(tr)

    def results(self):
        total_pnl = sum([t.profit for t in self.trades])
        return {"cash": self.cash, "closed_trades": len(self.trades), "pnl": total_pnl}
