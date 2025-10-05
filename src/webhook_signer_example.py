"""Example: HMAC webhook signer for broker-side integration.

This script demonstrates how a broker or external system can sign webhook payloads
using a shared secret. The signature is sent in the X-Hub-Signature-256 header.
"""
import hmac
import hashlib
import json
import requests
import os

WEBHOOK_URL = os.getenv('WEBHOOK_URL', 'http://localhost:8000/order-update')
WEBHOOK_HMAC_SECRET = os.getenv('WEBHOOK_HMAC_SECRET', 'demo_secret')

payload = {
    'order_id': 'K54321',
    'status': 'complete'
}

body = json.dumps(payload).encode('utf-8')
mac = hmac.new(WEBHOOK_HMAC_SECRET.encode('utf-8'), body, hashlib.sha256)
signature = mac.hexdigest()

headers = {
    'Content-Type': 'application/json',
    'X-Hub-Signature-256': signature
}

resp = requests.post(WEBHOOK_URL, data=body, headers=headers)
print('Status:', resp.status_code)
print('Response:', resp.text)
