import argparse
from .utils import setup_logging, load_env
from .data_fetcher import fetch_history
from .strategy import MACrossoverStrategy, RSIOLStrategy
from .plotter import plot_candles, add_indicator
from .kite_client import KiteClient
from .notifier import desktop_alert
import os


def demo(symbol: str = "RELIANCE.NS", simulated: bool = True):
    setup_logging()
    env = load_env()
    kc = KiteClient(simulated=simulated)
    df = fetch_history(symbol, period="1y", interval="1d")
    if df.empty:
        desktop_alert(f"No data for {symbol}")
        return
    strat = MACrossoverStrategy(short_window=20, long_window=50)
    sig = strat.generate(symbol, df)
    print("Signal:", sig.to_dict())
    fig = plot_candles(df, title=f"{symbol} price")
    ma20 = df['close'].rolling(20).mean()
    ma50 = df['close'].rolling(50).mean()
    add_indicator(fig, df, ma20, "MA20")
    add_indicator(fig, df, ma50, "MA50")
    fig.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--simulated", action="store_true")
    args = parser.parse_args()
    demo(args.symbol, simulated=args.simulated)


if __name__ == '__main__':
    main()
