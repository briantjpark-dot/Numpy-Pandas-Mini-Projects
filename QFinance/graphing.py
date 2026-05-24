import matplotlib.pyplot as plt
import numpy as np
from Universe import tickers
from Backend import data, add_signal, summarize, sharpe_ratio, backtest


def signal_data(df, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Price', color='black')
    plt.plot(df['SMA_10'], label='SMA 10', color='blue')
    plt.plot(df['SMA_30'], label='SMA 30', color='red')
    plt.title(f'{ticker} Price and SMA Signals')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid()
    plt.show()


def portfolio_performance(df, ticker):
    buy_and_hold = (1 + df['Daily Return']).cumprod()
    plt.figure(figsize=(14, 7))
    plt.plot(df['Cumulative'], label='Strategy', color='green')
    plt.plot(buy_and_hold, label='Buy & Hold', color='blue')
    plt.title(f'{ticker} Strategy vs Buy & Hold')
    plt.ylabel('Growth for $1 Invested')
    plt.xlabel('Date')
    plt.legend()
    plt.grid()
    plt.show()


for t in tickers:
    df = data(t)
    df = add_signal(df)
    summarize(df, t)
    df = backtest(df, t)
    signal_data(df, t)
    portfolio_performance(df, t)
