import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

data = rng.normal(loc = 0.0005, scale = 0.01, size = (3,20)).T
#.T flips 3 rows 20 columns to 3 columns 20 rows
#loc is mean, scale is std

df = pd.DataFrame(data, columns = ['AAPL', 'MSFT', 'GOOG'], index = pd.RangeIndex(20))

df['portfolio'] = df[['AAPL', 'MSFT', 'GOOG']].mean(axis=1)

df['week'] = df.index // 5

per_week = df.groupby('week')[['AAPL', 'MSFT', 'GOOG']].mean()

first_three = df[['AAPL', 'MSFT', 'GOOG']]

print(first_three.mean().idxmax())

# This script simulates 20 days of daily log returns for three stocks (AAPL, MSFT, GOOG) using
# NumPy's random number generator with a fixed seed for reproducibility. The returns are stored
# in a Pandas DataFrame, where we add a portfolio column representing the equal-weighted average
# return across all three stocks each day. We then create a week column by integer-dividing the
# row index by 5, grouping the 20 days into 4 weekly buckets. Using groupby we compute the mean
# return per stock per week, and finally identify which stock had the highest mean return overall
# across all 20 days using idxmax().