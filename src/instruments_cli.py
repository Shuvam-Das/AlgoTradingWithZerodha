"""CLI to download and save Kite instrument dump.

Usage:
  python -m src.instruments_cli --exchange NSE

This will attempt to use the kiteconnect package (and KITE_API_KEY from environment)
to fetch the instrument list and save to `.instruments.csv` in the workspace root.
If kiteconnect is not available or API key missing, you can provide a fallback CSV URL
with `--url`.
"""
import os
import csv
import argparse
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("atwz.instruments_cli")

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None


def save_instruments_to_csv(instruments: List[Dict], path: str = '.instruments.csv') -> None:
    if not instruments:
        raise ValueError("No instruments to save")
    fieldnames = list(instruments[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in instruments:
            writer.writerow(row)
    logger.info("Saved %d instruments to %s", len(instruments), path)


def fetch_using_kite(api_key: str, exchange: str = 'NSE') -> List[Dict]:
    if KiteConnect is None:
        raise RuntimeError('kiteconnect not installed')
    kite = KiteConnect(api_key=api_key)
    # kite.instruments can accept exchange or return full dump; behavior depends on kiteconnect version
    try:
        instruments = kite.instruments(exchange)
    except Exception:
        # some versions expect no args
        instruments = kite.instruments()
    # ensure instruments is a list of dicts
    return instruments


def fetch_from_url(url: str) -> List[Dict]:
    import requests  # type: ignore[import]
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    # assume CSV
    text = r.text.splitlines()
    rdr = csv.DictReader(text)
    return [row for row in rdr]


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exchange', default='NSE')
    parser.add_argument('--out', default='.instruments.csv')
    parser.add_argument('--url', help='Fallback CSV URL')
    args = parser.parse_args(argv)

    api_key = os.getenv('KITE_API_KEY')
    instruments = []
    if api_key and KiteConnect is not None:
        try:
            instruments = fetch_using_kite(api_key, args.exchange)
        except Exception as e:
            logger.warning('kite fetch failed: %s', e)

    if not instruments and args.url:
        try:
            instruments = fetch_from_url(args.url)
        except Exception as e:
            logger.error('fetch from url failed: %s', e)

    if not instruments:
        raise RuntimeError('Could not fetch instruments; provide KITE_API_KEY or --url')

    save_instruments_to_csv(instruments, path=args.out)


if __name__ == '__main__':
    main()
