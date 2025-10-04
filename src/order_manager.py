"""Order manager: handles placing orders, persistence and confirmation polling."""
import logging
import time
import uuid
from typing import Optional, Dict, Any, Callable, List
from .storage import init_db, save_order, update_order, get_order
from .kite_live import KiteLive
from .notifier import desktop_alert, notify_telegram, notify_telegram_from_env
from .storage import get_order_by_kite_id

logger = logging.getLogger("atwz.order_manager")


class OrderManager:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, simulated: bool = False, session_file: str = '.kite_session'):
        init_db()
        self.kite = KiteLive(api_key=api_key, api_secret=api_secret, simulated=simulated, session_file=session_file)
        # external update handlers (e.g. webhook will call process_update)
        self._external_handlers: List[Callable[[Dict[str, Any]], None]] = []

    def place_and_track(self, symbol: str, qty: int, side: str, price: Optional[float] = None, max_retries: int = 3, poll_interval: float = 2.0) -> Dict[str, Any]:
        local_id = str(uuid.uuid4())
        save_order(local_id, symbol, qty, side, price, status='created')
        try:
            res = self.kite.place_order(symbol, qty, side, price)
        except Exception:
            update_order(local_id, status='error')
            logger.exception("place_order failed")
            try:
                desktop_alert(f"Order placement failed for {symbol} {side} {qty}")
                notify_telegram_from_env(f"Order placement failed for {symbol} {side} {qty}")
            except Exception:
                logger.exception("notification failed after place_order error")
            return {'status': 'error'}

        # map result: if simulated returns Order dataclass, if live returns order id
        kite_order_id = None
        if isinstance(res, dict):
            kite_order_id = res.get('order_id')
        else:
            try:
                # simulated Order dataclass has id
                kite_order_id = getattr(res, 'id', None)
            except Exception:
                kite_order_id = None

        update_order(local_id, kite_order_id=kite_order_id, status='placed')

        # poll for confirmation if live
        retries = 0
        backoff = poll_interval
        while retries < max_retries:
            try:
                orders = self.kite.get_orders()
                # try to find matching order
                match = None
                for o in orders:
                    # handle dict or object
                    oid = o.get('order_id') if isinstance(o, dict) else getattr(o, 'id', None)
                    if oid and kite_order_id and str(oid) == str(kite_order_id):
                        match = o
                        break
                if match:
                    status = match.get('status') if isinstance(match, dict) else getattr(match, 'status', None)
                    update_order(local_id, status=status)
                    # notify any external handlers
                    for h in self._external_handlers:
                        try:
                            h({'local_id': local_id, 'kite_order_id': kite_order_id, 'status': status, 'kite_order': match})
                        except Exception:
                            logger.exception("external handler failed")
                    return {'status': 'confirmed', 'kite_order': match}
            except Exception:
                logger.exception("Polling orders failed")
            time.sleep(backoff)
            retries += 1
            backoff *= 2

        # if here, confirmation not found -> attempt cancel and mark unconfirmed
        try:
            if kite_order_id:
                logger.info("Attempting to cancel unconfirmed order %s", kite_order_id)
                self.kite.cancel_order(kite_order_id)
                update_order(local_id, status='cancelled')
                try:
                    desktop_alert(f"Order {kite_order_id} cancelled after no confirmation")
                    notify_telegram_from_env(f"Order {kite_order_id} cancelled after no confirmation")
                except Exception:
                    logger.exception("notification failed after cancel attempt")
                return {'status': 'cancelled', 'local_id': local_id}
        except Exception:
            logger.exception("Failed to cancel order")

        update_order(local_id, status='unconfirmed')
        return {'status': 'unconfirmed', 'local_id': local_id}

    def register_external_handler(self, fn):
        """Register a callable to receive order lifecycle updates from either polling or webhook receives.

        Handler will be called with a single dict argument describing the update.
        """
        self._external_handlers.append(fn)

    def process_external_update(self, payload: dict):
        """Process an incoming external order update (from webhook_receiver).

        The payload is expected to include a broker/kite order id in `order_id` or `kite_order_id`.
        We will find the matching local order and update status in storage, then call handlers.
        """
        kite_order_id = payload.get('order_id') or payload.get('kite_order_id') or payload.get('id')
        status = payload.get('status')
        if not kite_order_id:
            logger.warning("process_external_update called without kite_order_id")
            return
        existing = get_order_by_kite_id(str(kite_order_id))
        if existing:
            update_order(existing['local_id'], status=status)
            # call handlers
            for h in self._external_handlers:
                try:
                    h({'local_id': existing['local_id'], 'kite_order_id': kite_order_id, 'status': status, 'raw': payload})
                except Exception:
                    logger.exception("external handler failed")
