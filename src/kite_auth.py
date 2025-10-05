"""Development helper to perform Kite Connect login flow and exchange request token.

Usage (dev): set KITE_API_KEY and KITE_API_SECRET in env, run this Flask app, open
the login URL, login to Kite and allow, then the callback will store access token to
`.kite_session` file for the live adapter to pick up.

This helper is intentionally minimal and for development only.
"""

import os
import logging
import logging
from flask import Flask, redirect, request

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

logger = logging.getLogger("atwz.kite_auth")

app = Flask(__name__)

@app.route('/')
def index():
    api_key = os.getenv('KITE_API_KEY')
    if not api_key or KiteConnect is None:
        return "KITE_API_KEY missing or kiteconnect package not installed"
    try:
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()
        return redirect(login_url)
    except Exception as e:
        logger.exception("Invalid KITE_API_KEY or error in KiteConnect")
        return f"Error: {str(e)}. Please check your KITE_API_KEY."

@app.route('/callback')
def callback():
    api_key = os.getenv('KITE_API_KEY')
    api_secret = os.getenv('KITE_API_SECRET')
    request_token = request.args.get('request_token')
    if not request_token:
        return "No request_token in callback"
    kite = KiteConnect(api_key=api_key)
    try:
        data = kite.generate_session(request_token, api_secret)
        access_token = data.get('access_token')
        with open('.kite_session', 'w') as f:
            f.write(access_token)
        return "Access token saved. You can close this page."
    except Exception as e:
        logger.exception("Failed to generate session")
        return str(e)

def run_app(host='127.0.0.1', port=5000):
    if KiteConnect is None:
        raise RuntimeError('kiteconnect not installed')
    app.run(host=host, port=port)
