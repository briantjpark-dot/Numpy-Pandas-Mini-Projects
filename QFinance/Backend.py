import yfinance as yf
import pandas as pd
from Universe import tickers

# df stands for dataframe, used by pandas to store and manipulate tabular data.
# It is a 2-dimensional labeled data structure with columns of potentially different types,
# similar to a spreadsheet or SQL table.


def data(ticker, period="1y"):
    df = yf.Ticker(ticker).history(period=period)
    return df


def add_signal(df):
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_30'] = df['Close'].rolling(30).mean()

    df['MA_Signal'] = 0
    df.loc[df['SMA_10'] > df['SMA_30'], 'MA_Signal'] = 1
    df.loc[df['SMA_10'] < df['SMA_30'], 'MA_Signal'] = -1

    df['Daily Return'] = df['Close'].pct_change()

    return df


def summarize(df, ticker):
    latest = df.iloc[-1]
    signal = latest['MA_Signal']
    label = "BULLISH" if signal == 1 else "BEARISH" if signal == -1 else "HOLD"

    print(f"\n-- {ticker} --")
    print(f"Price: ${latest['Close']:.2f}")
    print(f"SMA_10: ${latest['SMA_10']:.2f}")
    print(f"SMA_30: ${latest['SMA_30']:.2f}")
    print(f"Signal: {label}")


def sharpe_ratio(df, risk_free_rate=0.0457):
    excess_return = df['Daily Return'] - (risk_free_rate / 252)
    sharpe = excess_return.mean() / excess_return.std() * (252 ** 0.5)
    return round(sharpe, 3)


def backtest(df, ticker, risk_free_rate=0.0457):
    df = df.copy()
    # Shift signal by 1 to avoid lookahead bias; long-only (clip to 0)
    df['Position'] = df['MA_Signal'].shift(1).fillna(0).clip(lower=0)
    df['Strategy Return'] = df['Position'] * df['Daily Return']
    df['Cumulative'] = (1 + df['Strategy Return']).cumprod()

    total_return = df['Cumulative'].iloc[-1] - 1

    rolling_max = df['Cumulative'].cummax()
    drawdown = (df['Cumulative'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    excess = df['Strategy Return'] - (risk_free_rate / 252)
    sharpe = round(excess.mean() / excess.std() * (252 ** 0.5), 3)

    print(f"\n=== Backtest: {ticker} ===")
    print(f"Total Return:  {total_return:.2%}")
    print(f"Max Drawdown:  {max_drawdown:.2%}")
    print(f"Sharpe Ratio:  {sharpe}")
    print(f"Strategy: {total_return:.1%}")

    return df


for t in tickers:
    df = data(t)
    df = add_signal(df)
    summarize(df, t)
    backtest(df, t)
   