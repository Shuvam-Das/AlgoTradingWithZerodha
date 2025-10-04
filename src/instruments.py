"""Instrument mapping utilities.

This module helps map trading symbols (like RELIANCE) to instrument tokens used by
Kite's LTP API. It can load the instrument dump CSV (downloaded from Kite) or accept
a small mapping provided by the user.
"""
import csv
import os
from typing import Dict, Optional

INSTRUMENT_CSV = '.instruments.csv'


def load_instruments(path: str = INSTRUMENT_CSV) -> Dict[str, Dict]:
    """Load an instrument dump CSV into a mapping keyed by exchange:tradingsymbol and token."""
    instruments: Dict[str, Dict] = {}
    if not os.path.exists(path):
        return instruments
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            # expected fields may include instrument_token, exchange, tradingsymbol
            key = f"{row.get('exchange')}:{row.get('tradingsymbol')}"
            instruments[key] = row
    return instruments


def find_instrument_token(instruments: Dict[str, Dict], exchange: str, tradingsymbol: str) -> Optional[str]:
    key = f"{exchange}:{tradingsymbol}"
    row = instruments.get(key)
    if not row:
        return None
    # instrument_token or token
    return row.get('instrument_token') or row.get('token')


def get_instrument_details(instruments: Dict[str, Dict], exchange: str, tradingsymbol: str) -> Optional[Dict]:
    key = f"{exchange}:{tradingsymbol}"
    row = instruments.get(key)
    if not row:
        return None
    # normalize common fields
    details = {}
    details['instrument_token'] = row.get('instrument_token') or row.get('token')
    # lot size may be present as 'lot_size' or 'lot_size' in various dumps
    ls = row.get('lot_size') or row.get('lotSize') or row.get('lot')
    try:
        details['lot_size'] = int(ls) if ls else None
    except Exception:
        details['lot_size'] = None
    # margin or margin_rate may be present
    mr = row.get('margin') or row.get('margin_rate')
    try:
        details['margin_rate'] = float(mr) if mr else None
    except Exception:
        details['margin_rate'] = None
    return details
