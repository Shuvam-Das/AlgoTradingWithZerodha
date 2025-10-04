import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from instruments import load_instruments, find_instrument_token


def test_load_and_find(tmp_path):
    p = tmp_path / "instruments.csv"
    p.write_text('exchange,tradingsymbol,instrument_token\nNSE,RELIANCE,12345\n')
    instruments = load_instruments(str(p))
    assert 'NSE:RELIANCE' in instruments
    token = find_instrument_token(instruments, 'NSE', 'RELIANCE')
    assert token == '12345'
