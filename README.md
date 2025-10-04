# AlgoTradingWithZerodha
Algo Trading with Zerodha using python

This repository contains a scaffold prototype of an advanced trading bot. See `src/README.md` for details.

Quick tools
- Download Kite instrument dump (requires KITE_API_KEY and kiteconnect):

	python -m src.instruments_cli --exchange NSE

- Fetch Kite margin details (requires `.kite_session` or KITE_API_KEY):

	python -m src.margin_cli

- Run the Streamlit order-history UI:

	streamlit run src/ui_streamlit.py

Environment variables
- KITE_API_KEY, KITE_API_SECRET (for Kite connect)
- SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS (email alerts)
- TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (Telegram alerts)
- WEBHOOK_HMAC_SECRET (shared secret for webhook HMAC verification)

## Webhook Integration (HMAC-Signed)

This project supports secure broker/external webhook callbacks using HMAC signatures. All sensitive endpoints (order updates, trading signals, admin actions) require a valid HMAC signature header.

**How it works:**
- The sender (broker/external system) signs the JSON payload using a shared secret and includes the signature in the `X-Hub-Signature-256` header.
- The receiver verifies the signature before processing the request. Invalid or missing signatures are logged and trigger security alerts (e.g., Telegram).

**Example sender (see `src/webhook_signer_example.py`):**
```python
import hmac, hashlib, json, requests, os
WEBHOOK_URL = os.getenv('WEBHOOK_URL', 'http://localhost:8000/order-update')
WEBHOOK_HMAC_SECRET = os.getenv('WEBHOOK_HMAC_SECRET', 'demo_secret')
payload = {'order_id': 'K54321', 'status': 'complete'}
body = json.dumps(payload).encode('utf-8')
mac = hmac.new(WEBHOOK_HMAC_SECRET.encode('utf-8'), body, hashlib.sha256)
signature = mac.hexdigest()
headers = {'Content-Type': 'application/json', 'X-Hub-Signature-256': signature}
resp = requests.post(WEBHOOK_URL, data=body, headers=headers)
print('Status:', resp.status_code)
```

**Receiver security:**
- All sensitive endpoints (webhook, /predict, /train, /backtest) require HMAC signature.
- Failed signature attempts are logged and trigger alerts via Telegram (if configured).

**To enable:**
- Set `WEBHOOK_HMAC_SECRET` in your environment (same value for sender and receiver).
- Optionally configure Telegram alerts for security notifications.

**See also:**
- `src/webhook_receiver.py` (receiver implementation)
- `src/webhook_signer_example.py` (example sender)
- `tests/test_webhook_hmac.py` (test coverage)

CI / Auto-update
- The repository includes a GitHub Actions workflow `.github/workflows/update-instruments.yml` which runs daily (02:00 UTC) and can be dispatched manually.
- The workflow will try to fetch the instrument dump using `src.instruments_cli`. You can provide a fallback URL using the `INSTRUMENTS_URL` secret in the repository settings (recommended if you host a trusted mirror).
- To enable Kite-based fetching, set `KITE_API_KEY` and relevant secrets in GitHub secrets and ensure the action can authenticate (you may need to provide an access token via secrets).
