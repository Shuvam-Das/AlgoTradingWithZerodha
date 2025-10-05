from typing import Dict, Any, List, Optional
import os
import uuid
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("atwz.kite")


@dataclass
class Order:
    symbol: str
    qty: int
    side: str  # 'buy' or 'sell'
    price: Optional[float] = None
    status: str = "created"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class KiteClient:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, simulated: bool = True) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.simulated: bool = bool(simulated or (api_key is None or api_secret is None))
        self.orders: Dict[str, Order] = {}
        logger.info(f"KiteClient simulated={self.simulated}")

    def place_order(self, symbol: str, qty: int, side: str, price: Optional[float] = None) -> Order:
        order = Order(symbol=symbol, qty=qty, side=side, price=price)
        order.status = "filled" if self.simulated else "pending"
        self.orders[order.id] = order
        logger.info(f"Placed order {order}")
        return order

    def get_orders(self) -> List[Order]:
        return list(self.orders.values())

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = "cancelled"
            return True
        return False
