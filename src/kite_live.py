"""Kite Connect live adapter with safe simulated fallback.

This module attempts to import the official `kiteconnect` package and provide a
thin wrapper matching KiteClient's interface. If the package or API keys are not
available, it falls back to the simulated client.
"""
import logging
import os
from typing import Optional

from .kite_client import KiteClient as SimKite
from .instruments import load_instruments, find_instrument_token

logger = logging.getLogger("atwz.kite_live")

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None


class KiteLive:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, request_token: Optional[str] = None, simulated: bool = False, session_file: str = ".kite_session"):
        self.simulated = simulated or (KiteConnect is None or not api_key or not api_secret)
        self.session_file = session_file
        if self.simulated:
            logger.info("KiteLive running in simulated mode")
            self.client = SimKite(api_key, api_secret, simulated=True)
            return

        self.kite = KiteConnect(api_key=api_key)
        # load stored access token if exists
        if os.path.exists(session_file):
            try:
                data = {}
                with open(session_file, 'r') as f:
                    line = f.read().strip()
                    access_token = line
                self.kite.set_access_token(access_token)
                logger.info("Loaded Kite session from %s", session_file)
                return
            except Exception:
                logger.exception("Failed to load kite session")

        # Exchange request token for access token
        if request_token:
            try:
                data = self.kite.generate_session(request_token, api_secret)
                access_token = data.get('access_token')
                if access_token:
                    self.kite.set_access_token(access_token)
                    with open(session_file, 'w') as f:
                        f.write(access_token)
                    logger.info("Saved access token to %s", session_file)
            except Exception:
                logger.exception("Failed to generate kite session from request token")

    def place_order(self, symbol: str, qty: int, side: str, price: Optional[float] = None):
        if self.simulated:
            return self.client.place_order(symbol, qty, side, price)
        try:
            # map our simplified order into kite order params (example: market order)
            order_type = "MARKET"
            transaction_type = "BUY" if side.lower() == 'buy' else 'SELL'
            params = dict(tradingsymbol=symbol, exchange="NSE", transaction_type=transaction_type, quantity=qty, order_type=order_type, product="MIS")
            if price:
                params['price'] = price
            order_id = self.kite.place_order(**params)
            logger.info("Placed live order %s", order_id)
            # kite.place_order returns order id string in many versions; return dict
            return {"order_id": order_id}
        except Exception:
            logger.exception("Live place_order failed")
            raise

    def get_orders(self):
        if self.simulated:
            return self.client.get_orders()
        try:
            # normalize to list of dicts
            orders = self.kite.orders()
            return orders
        except Exception:
            logger.exception("get_orders failed")
            return []

    def ltp(self, symbol: str) -> Optional[float]:
        if self.simulated:
            # In simulated mode return a dummy price
            return 1.0
        try:
            # Prefer using instrument token mapping if instrument dump is available
            instruments = load_instruments()
            token = None
            if instruments:
                token = find_instrument_token(instruments, 'NSE', symbol)
            if token:
                data = self.kite.ltp({token: token}) if isinstance(token, str) else self.kite.ltp(token)
                # data example: {'instrument_token': {'last_price': 2500.0}}
                key = next(iter(data.keys()))
                return float(data[key].get('last_price'))
            # fallback to exchange:symbol style
            data = self.kite.ltp(f"NSE:{symbol}")
            key = next(iter(data.keys()))
            return float(data[key].get('last_price'))
        except Exception:
            logger.exception("ltp fetch failed")
            return None

    def cancel_order(self, order_id: str) -> bool:
        if self.simulated:
            return self.client.cancel_order(order_id)
        try:
            self.kite.cancel_order(order_id)
            return True
        except Exception:
            logger.exception("cancel_order failed")
            return False
