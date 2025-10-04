import os
import sys
# Ensure workspace root is on path so 'src' package imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.storage import init_db, save_order, get_order, update_order
from src.order_manager import OrderManager


def test_storage(tmp_path):
    db = tmp_path / "orders.db"
    # initialize DB at path
    init_db(str(db))
    save_order('local-1', 'RELIANCE', 1, 'buy', 100.0, status='created', path=str(db))
    o = get_order('local-1', path=str(db))
    assert o['symbol'] == 'RELIANCE'
    update_order('local-1', status='placed', path=str(db))
    o2 = get_order('local-1', path=str(db))
    assert o2['status'] == 'placed'


def test_order_manager_simulated():
    om = OrderManager(simulated=True)
    res = om.place_and_track('RELIANCE', 1, 'buy', price=1.0)
    assert res['status'] in ('confirmed', 'unconfirmed', 'error')
