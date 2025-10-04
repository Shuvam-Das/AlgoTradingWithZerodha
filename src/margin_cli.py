"""CLI to fetch margin details using Kite and print them."""
import argparse
from .margin_kite import fetch_margin_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', default='.kite_session')
    args = parser.parse_args()
    data = fetch_margin_details(session_file=args.session)
    print(data)


if __name__ == '__main__':
    main()
