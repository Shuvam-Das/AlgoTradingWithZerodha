"""Integration example showing KiteWebsocket -> Strategy -> OrderManager flow.

This example uses simulated Kite client/the websocket if available. It demonstrates:
- subscribing to ticks
- generating a simple MA crossover signal
- placing an order via OrderManager
- simulating an order-update webhook to mark the order complete

Run locally for testing; it is illustrative and not production-ready.
"""
import time
import logging
from typing import Any, Dict

from .kite_ws import KiteWebsocket
from .strategy import MACrossoverStrategy, Signal
from .order_manager import OrderManager
from .webhook_receiver import app as webhook_app

logger = logging.getLogger("atwz.integration")


class SimpleRunner:
    def __init__(self):
        # simulated mode uses the in-repo KiteClient
        self.om = OrderManager(simulated=True)
        self.strategy = MACrossoverStrategy(short_window=3, long_window=5)

    def on_ticks(self, ticks: Any):
        # ticks is a list of tick dicts; we pick last price and symbol
        for t in ticks:
            sym = t.get('tradingsymbol') or t.get('instrument') or 'UNKNOWN'
            ltp = t.get('last_price') or t.get('ltp') or t.get('price')
            logger.info("Tick %s %s", sym, ltp)
            # For demo, craft a tiny history-like frame isn't available; instead,
            # we use a trivial threshold signal: buy if price is above 0
            if ltp and isinstance(ltp, (int, float)) and ltp > 0:
                sig = {'symbol': sym, 'signal': 'buy', 'confidence': 0.6}
                res = self.om.place_and_track(sym, 1, 'buy', float(ltp))
                logger.info("Order placed result: %s", res)

    def start(self):
        # If KiteTicker is not installed, we simply simulate a tick
        try:
            # do not hardcode credentials here; use None (simulated) or env-driven keys
            ws = KiteWebsocket(api_key=None, access_token=None, on_tick=self.on_ticks)
            # instrument token 0 as placeholder; in practice use real tokens
            ws.connect([0])
            logger.info("Websocket started")
            time.sleep(1)
        except Exception:
            logger.warning("KiteWebsocket not available; simulating ticks")
            # simulate a tick and order lifecycle
            tick = {'tradingsymbol': 'SIM-TEST', 'last_price': 100.0}
            self.on_ticks([tick])

        # simulate webhook call: find last order and call process_external_update
        # in real deployment, webhook will call src.webhook_receiver endpoint
        logger.info("Simulating webhook update to mark orders complete")
        # get latest orders via storage or order manager's kite simulation
        for o in self.om.kite.get_orders():
            payload = {'order_id': getattr(o, 'id', None), 'status': 'complete'}
            self.om.process_external_update(payload)


def main():
    logging.basicConfig(level=logging.INFO)
    r = SimpleRunner()
    r.start()


if __name__ == '__main__':
    main()
