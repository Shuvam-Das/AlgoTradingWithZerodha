import hmac
import hashlib
import json
import os
import pytest

# Skip test if FastAPI is not available in the environment
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient
from src.webhook_receiver import app, WEBHOOK_HMAC_SECRET
from src.storage import init_db, get_order_by_kite_id, save_order

def sign_payload(payload: dict, secret: str) -> str:
    body = json.dumps(payload).encode('utf-8')
    mac = hmac.new(secret.encode('utf-8'), body, hashlib.sha256)
    return mac.hexdigest()

def test_hmac_webhook(tmp_path, monkeypatch):
    db = tmp_path / 'orders.db'
    import src.storage as storage
    storage.DB_PATH = str(db)
    init_db(path=storage.DB_PATH)
    local_id = 'local-2'
    kite_id = 'K54321'
    save_order(local_id, 'TEST-XYZ', 1, 'buy', 100.0, status='placed', path=storage.DB_PATH)
    storage.update_order(local_id, kite_order_id=kite_id, status='placed', path=storage.DB_PATH)
    client = TestClient(app)
    payload = {'order_id': kite_id, 'status': 'complete'}
    signature = sign_payload(payload, WEBHOOK_HMAC_SECRET)
    r = client.post('/order-update', json=payload, headers={'X-Hub-Signature-256': signature})
    assert r.status_code == 200
    found = get_order_by_kite_id(kite_id, path=storage.DB_PATH)
    assert found is not None
    assert found['status'] == 'complete'
    # Test with invalid signature
    r2 = client.post('/order-update', json=payload, headers={'X-Hub-Signature-256': 'bad_signature'})
    assert r2.status_code == 401
