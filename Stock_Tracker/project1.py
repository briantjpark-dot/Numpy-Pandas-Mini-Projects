import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = yf.download(["AAPL", "MSFT", "TSLA", "ASML"], period="1mo")

weightings = np.array([0.4, 0.3, 0.2, 0.1])

close = data['Close']

# Count missing values per ticker to understand data quality
empty = close.isnull().sum()

# Fill any missing values forward (carries the last known price forward)
close = close.ffill()

# Wasn't in past exercises but collapses on a weekly basis, built into pandas
close_weekly = close.resample("W").mean()

# % change
per_change = close_weekly.pct_change()

per_change = per_change.dropna()
# alternatively could do .[1:] but that's because we know the first row will all be Na

means = per_change.mean()

standard_deviation = per_change.std()
# Standard deviation measures how much returns bounce around each week — high std = high risk.
# Sharpe divides mean return by std to show return per unit of risk. sqrt(52) scales both to annual.

weighted_change = np.multiply(weightings, per_change)
# portfolio returns weighted by our hardcode on ln 7

portfolio_returns = weighted_change.sum(axis=1)
# "axis=1" sums across the ticker columns for each week

cum_portfolio = (1+portfolio_returns).cumprod()
# cumprod stands for cumulative product, like where you multiply by the previous value

annual_volatility = standard_deviation * np.sqrt(52)
# Annualized volatility per ticker — weekly std scaled up to a full year (52 weeks)

correlation = per_change.corr()
# Correlation matrix — shows how similarly each pair of tickers moves week to week --> pretty, confused though

Sharpe = means / standard_deviation * (52 ** 0.5)

portfolio_volatility = portfolio_returns.std() * np.sqrt(52)
# Same volatility calculation but for the whole portfolio as one combined return series

portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(52)
# Sharpe ratio for the portfolio — how much return you're getting per unit of risk across all holdings

cum_tickers = (1 + per_change).cumprod()

cum_tickers.plot()
cum_portfolio.plot(label="Portfolio")

plt.title("Final Graph")
plt.xlabel("Date")
plt.ylabel("Growth per $1")
plt.legend()
plt.show()

summary = pd.DataFrame({
    "total_return": means,
    "annual_volatility": annual_volatility,
    "sharpe": Sharpe
})

summary.to_csv("portfolio_summary.csv")
