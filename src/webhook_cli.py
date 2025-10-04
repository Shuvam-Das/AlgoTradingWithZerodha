"""Utility to run the webhook receiver and wire it into the OrderManager.

Run with: python -m src.webhook_cli
"""
import logging
from .webhook_receiver import app, register_handler
from .order_manager import OrderManager

logger = logging.getLogger("atwz.webhook_cli")


def _on_update(payload):
    # simple handler that uses OrderManager to process updates
    try:
        # For this lightweight CLI we create an OrderManager in simulated mode
        om = OrderManager(simulated=True)
        om.process_external_update(payload)
    except Exception:
        logger.exception("failed to process incoming webhook")


def main():
    register_handler(_on_update)
    # start uvicorn programmatically for convenience
    import uvicorn
    # Bind to localhost by default; for production override HOST env var and use TLS/reverse proxy
    host = os.getenv('WEBHOOK_HOST', '127.0.0.1')
    port = int(os.getenv('WEBHOOK_PORT', '8000'))
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
