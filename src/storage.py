"""Simple SQLite storage for orders and trades."""
import sqlite3
import threading
import time
from typing import Optional, Dict, Any

DB_PATH = '.orders.db'
_lock = threading.Lock()


def _get_conn(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str = DB_PATH):
    with _lock:
        conn = _get_conn(path)
        cur = conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            local_id TEXT,
            kite_order_id TEXT,
            symbol TEXT,
            qty INTEGER,
            side TEXT,
            price REAL,
            status TEXT,
            created_at REAL,
            updated_at REAL
        )
        ''')
        conn.commit()
        conn.close()


def save_order(local_id: str, symbol: str, qty: int, side: str, price: float | None, status: str = 'created', path: str = DB_PATH):
    ts = time.time()
    with _lock:
        conn = _get_conn(path)
        cur = conn.cursor()
        cur.execute('''INSERT INTO orders(local_id,kite_order_id,symbol,qty,side,price,status,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?)''', (local_id, None, symbol, qty, side, price, status, ts, ts))
        conn.commit()
        conn.close()


def update_order(local_id: str, kite_order_id: Optional[str] = None, status: Optional[str] = None, path: str = DB_PATH):
    ts = time.time()
    with _lock:
        conn = _get_conn(path)
        cur = conn.cursor()
        if kite_order_id is not None and status is not None:
            cur.execute('UPDATE orders SET kite_order_id=?, status=?, updated_at=? WHERE local_id=?', (kite_order_id, status, ts, local_id))
        elif kite_order_id is not None:
            cur.execute('UPDATE orders SET kite_order_id=?, updated_at=? WHERE local_id=?', (kite_order_id, ts, local_id))
        elif status is not None:
            cur.execute('UPDATE orders SET status=?, updated_at=? WHERE local_id=?', (status, ts, local_id))
        conn.commit()
        conn.close()


def get_order(local_id: str, path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = _get_conn(path)
        cur = conn.cursor()
        cur.execute('SELECT * FROM orders WHERE local_id=?', (local_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None


def get_order_by_kite_id(kite_order_id: str, path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Find an order by the broker/kite order id."""
    with _lock:
        conn = _get_conn(path)
        cur = conn.cursor()
        cur.execute('SELECT * FROM orders WHERE kite_order_id=?', (kite_order_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None
