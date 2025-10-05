"""Webhook receiver to accept broker order update callbacks and dispatch to handlers.

Provides a small FastAPI app with a POST /order-update endpoint. The endpoint expects
JSON payloads containing at least `order_id` (broker order id) and `status`. Integrators
can run this via `uvicorn src.webhook_receiver:app` behind a TLS-enabled reverse proxy or
use a tunneling service for local testing.
"""

from fastapi import FastAPI, Request, HTTPException
from typing import Callable, List, Dict, Any
import logging
import hmac
import hashlib
import os

from .storage import update_order, get_order_by_kite_id

logger = logging.getLogger("atwz.webhook")

app = FastAPI()

# Registered handlers will be called with the parsed update dict
_registered_handlers: List[Callable[[Dict[str, Any]], None]] = []

# HMAC secret (set via env or config)
WEBHOOK_HMAC_SECRET = os.getenv('WEBHOOK_HMAC_SECRET', 'demo_secret')

def register_handler(fn: Callable[[Dict[str, Any]], None]):
    """Register a callback to be called on every order update received via webhook."""
    _registered_handlers.append(fn)


def verify_hmac_signature(body: bytes, signature: str, secret: str) -> bool:
    mac = hmac.new(secret.encode('utf-8'), body, hashlib.sha256)
    expected = mac.hexdigest()
    return hmac.compare_digest(expected, signature.lower())

# FastAPI dependency for HMAC verification
from fastapi import Depends
from fastapi import Request
from fastapi import HTTPException
def hmac_signature_required(secret: str = WEBHOOK_HMAC_SECRET):
    async def dependency(request: Request):
        signature = request.headers.get('X-Hub-Signature-256') or request.headers.get('X-HMAC-Signature')
        body = await request.body()
        if not signature:
            logger.error(f"Missing HMAC signature header from {request.client.host if request.client else 'unknown'}")
            try:
                from .notifier import notify_telegram_from_env
                notify_telegram_from_env(f"SECURITY ALERT: Missing HMAC signature header on {request.url.path} from {request.client.host if request.client else 'unknown'}")
            except Exception:
                logger.exception("Failed to send alert for missing HMAC signature")
            raise HTTPException(status_code=401, detail="Missing HMAC signature header")
        if not verify_hmac_signature(body, signature, secret):
            logger.error(f"Invalid HMAC signature from {request.client.host if request.client else 'unknown'} on {request.url.path}")
            try:
                from .notifier import notify_telegram_from_env
                notify_telegram_from_env(f"SECURITY ALERT: Invalid HMAC signature on {request.url.path} from {request.client.host if request.client else 'unknown'}")
            except Exception:
                logger.exception("Failed to send alert for invalid HMAC signature")
            raise HTTPException(status_code=401, detail="Invalid HMAC signature")
    return Depends(dependency)

@app.post('/order-update')
async def order_update(request: Request):
    # HMAC verification
    signature = request.headers.get('X-Hub-Signature-256') or request.headers.get('X-HMAC-Signature')
    body = await request.body()
    if not signature:
        logger.error(f"Missing HMAC signature header from {request.client.host if request.client else 'unknown'}")
        try:
            from .notifier import notify_telegram_from_env
            notify_telegram_from_env(f"SECURITY ALERT: Missing HMAC signature header on /order-update from {request.client.host if request.client else 'unknown'}")
        except Exception:
            logger.exception("Failed to send alert for missing HMAC signature")
        raise HTTPException(status_code=401, detail="Missing HMAC signature header")
    if not verify_hmac_signature(body, signature, WEBHOOK_HMAC_SECRET):
        logger.error(f"Invalid HMAC signature from {request.client.host if request.client else 'unknown'} on /order-update")
        try:
            from .notifier import notify_telegram_from_env
            notify_telegram_from_env(f"SECURITY ALERT: Invalid HMAC signature on /order-update from {request.client.host if request.client else 'unknown'}")
        except Exception:
            logger.exception("Failed to send alert for invalid HMAC signature")
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Minimal validation
    kite_order_id = payload.get('order_id') or payload.get('kite_order_id') or payload.get('id')
    status = payload.get('status')

    if not kite_order_id or not status:
        raise HTTPException(status_code=400, detail="Missing order_id or status")

    # try to update storage if an order with this kite_order_id exists
    try:
        existing = get_order_by_kite_id(str(kite_order_id))
        if existing:
            # find local_id and update
            update_order(existing['local_id'], status=status)
        # dispatch to registered handlers
        for h in _registered_handlers:
            try:
                h(payload)
            except Exception:
                logger.exception("handler failed")
    except Exception:
        logger.exception("Failed to process order update")

    return {"status": "ok"}


if __name__ == '__main__':
    # Quick local runner for manual testing
    import uvicorn
    host = os.getenv('WEBHOOK_HOST', '127.0.0.1')
    port = int(os.getenv('WEBHOOK_PORT', '8000'))
    uvicorn.run(app, host=host, port=port)
