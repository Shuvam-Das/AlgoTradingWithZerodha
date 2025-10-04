"""Wrapper for Kite margin APIs.

Provides a small helper to request margin and portfolio details from Kite.
Requires kiteconnect and a valid access token in `.kite_session` or environment.
"""
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger("atwz.margin_kite")

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None


def get_kite_client(api_key: Optional[str] = None, session_file: str = '.kite_session') -> 'KiteConnect':
    if KiteConnect is None:
        raise RuntimeError('kiteconnect not installed')
    api_key = api_key or os.getenv('KITE_API_KEY')
    if not api_key:
        raise RuntimeError('KITE_API_KEY is required')
    k = KiteConnect(api_key=api_key)
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            token = f.read().strip()
            k.set_access_token(token)
    return k


def fetch_margin_details(api_key: Optional[str] = None, session_file: str = '.kite_session') -> Dict[str, Any]:
    k = get_kite_client(api_key, session_file)
    try:
        return k.margins()
    except Exception:
        logger.exception('Failed to fetch margins')
        raise
