"""Execution engine: takes signals and places orders via KiteLive (or simulated client).

Features:
- basic position sizing based on available cash
- safety checks (max order size, symbol validation)
- dry-run simulated mode
"""
import logging
from typing import Any, Dict, Optional
from .order_manager import OrderManager
from .kite_live import KiteLive
from .security import validate_symbol, read_api_keys_from_env
from .margin import required_margin_equity, has_buying_power
from .margin_kite import fetch_margin_details
from .instruments import load_instruments, get_instrument_details
from .margin import required_margin_fno_with_lot

logger = logging.getLogger("atwz.executor")


class Executor:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, simulated: bool = False, cash: float = 100000.0) -> None:
        env = read_api_keys_from_env()
        api_key = api_key or env.get('KITE_API_KEY')
        api_secret = api_secret or env.get('KITE_API_SECRET')
        self.cash = cash
        self.order_manager = OrderManager(api_key=api_key, api_secret=api_secret, simulated=simulated)
        self.client = self.order_manager.kite

    def size_order(self, price: float, risk_per_trade: float = 0.01) -> int:
        # simple sizing: use risk_per_trade fraction of cash to buy
        if price <= 0:
            return 0
        allocation = self.cash * risk_per_trade
        qty = int(allocation // price)
        return max(qty, 0)

    def size_order_with_margin(self, symbol: str, price: float, side: str, asset_class: str = 'equity', risk_per_trade: float = 0.01) -> int:
        """Size using Kite margins when available (for F&O uses margin-based buying power).

        asset_class: 'equity' or 'fno'
        """
        # If in simulated mode or no kite margin API available, fallback to simple sizing
        try:
            margins = fetch_margin_details()
        except Exception:
            return self.size_order(price, risk_per_trade)

        # margins is expected to contain 'net' or 'equity' fields depending on Kite's response
        # We will attempt to find daywise available cash or equity
        available_cash = None
        if isinstance(margins, dict):
            # Kite returns different structures; try common keys
            if 'equity' in margins:
                eq = margins.get('equity')
                available_cash = eq.get('available_cash') if isinstance(eq, dict) else None
            if available_cash is None and 'net' in margins:
                # fallback
                available_cash = margins.get('net', {}).get('available') if isinstance(margins.get('net'), dict) else None
        if available_cash is None:
            # fallback
            return self.size_order(price, risk_per_trade)

        if asset_class.lower() == 'equity':
            allocation = float(available_cash) * risk_per_trade
            return max(int(allocation // price), 0)
        else:
            # For F&O, use instrument dump to find lot_size and margin_rate
            instruments = load_instruments()
            details = get_instrument_details(instruments, 'NSE', symbol)
            if details and details.get('lot_size') and details.get('margin_rate'):
                lot = int(details['lot_size'])
                rate = float(details['margin_rate'])
                # margin required for one lot
                one_lot_margin = required_margin_fno_with_lot(price, lot, rate)
                if one_lot_margin <= 0:
                    return 0
                # how many lots can we buy with available_cash * risk_per_trade
                allocation = float(available_cash) * risk_per_trade
                lots = int(allocation // one_lot_margin)
                return max(lots * lot, 0)
            # fallback to naive notional-based sizing
            return max(int(float(available_cash) // price), 0)

    def execute_signal(self, signal: Dict[str, Any], price: Optional[float] = None) -> Dict[str, Any]:
        symbol = signal.get('symbol')
        action = signal.get('signal')
        confidence = signal.get('confidence', 0.5)
        if not isinstance(symbol, str) or not validate_symbol(symbol):
            raise ValueError("Invalid symbol")
        if action not in ('buy', 'sell', 'hold'):
            raise ValueError("Unknown action")
        if action == 'hold':
            return {'status': 'hold'}
        # determine price (market fallback)
        # fetch price automatically via LTP if not provided
        if price is None:
            price = self.client.ltp(symbol)
            if not isinstance(price, (int, float)):
                return {'status': 'skipped', 'reason': 'no-price'}
        qty = self.size_order(price)
        if qty <= 0:
            return {'status': 'skipped', 'reason': 'not enough cash or zero size'}
        # check buying power
        req = required_margin_equity(price, qty)
        if not has_buying_power(self.cash, req):
            return {'status': 'skipped', 'reason': 'insufficient buying power', 'required': req}
        res = self.order_manager.place_and_track(symbol, qty, action, price)
        logger.info("Executed %s %s x %s => %s", action, qty, symbol, res.get('status'))
        return res
