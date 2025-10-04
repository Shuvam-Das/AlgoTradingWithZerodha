import time
import pytest

# Skip test if FastAPI is not available in the environment
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from src.webhook_receiver import app
from src.storage import init_db, get_order_by_kite_id, save_order


def test_order_update_webhook(tmp_path, monkeypatch):
    # use a clean DB file in tmp_path
    db = tmp_path / 'orders.db'
    # monkeypatch DB path in storage module
    import src.storage as storage
    storage.DB_PATH = str(db)
    init_db(path=storage.DB_PATH)

    # save a dummy order with a kite id
    local_id = 'local-1'
    kite_id = 'K12345'
    save_order(local_id, 'TEST-ABC', 1, 'buy', 100.0, status='placed', path=storage.DB_PATH)
    # manually update kite_order_id to simulate placement
    storage.update_order(local_id, kite_order_id=kite_id, status='placed', path=storage.DB_PATH)

    client = TestClient(app)
    payload = {'order_id': kite_id, 'status': 'complete'}
    r = client.post('/order-update', json=payload)
    assert r.status_code == 200

    # allow small delay and then verify storage
    time.sleep(0.1)
    found = get_order_by_kite_id(kite_id, path=storage.DB_PATH)
    assert found is not None
    assert found['status'] == 'complete'
