import plotly.graph_objects as go
import pandas as pd


def plot_candles(df: pd.DataFrame, title: str = "Chart") -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    fig.update_layout(title=title)
    return fig

def add_indicator(fig: go.Figure, df: pd.DataFrame, series: pd.Series, name: str):
    fig.add_trace(go.Scatter(x=df.index, y=series, mode='lines', name=name))
    return fig
