
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


ticker = "^GSPC"
data = yf.download(ticker, start="2000-01-01", end="2026-06-01")

# Newer yfinance versions return "MultiIndex" columns like ('Close', '^GSPC')
# instead of plain 'Close' -- this flattens it back to plain column names so
# the rest of the script (and you, reading it) doesn't have to deal with that.
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# What did we get? A DataFrame -- think of it like a spreadsheet with
# dates as the row labels (the "index") and price info as columns.
print(data.head())   # first 5 rows
print(data.tail())   # last 5 rows
print(data.shape)    # (number of rows, number of columns)
print(data.columns)  # column names: Open, High, Low, Close, Volume (Adj Close folded into Close by default now)

# %% --------------------------------------------------------------------
# STEP 2: Save the raw data so we never have to re-download it
# --------------------------------------------------------------------
# This goes in data/raw/ in your repo. Saving raw data untouched is a
# research best practice -- you can always reproduce every later step
# from this file alone.

data.to_csv("sp500_raw.csv")
print("Saved raw data.")

# %% --------------------------------------------------------------------
# STEP 3: Basic exploration
# --------------------------------------------------------------------
# .describe() gives count, mean, std, min, max, and percentiles for each
# numeric column -- a fast way to sanity-check the data (e.g. is the min
# price negative? that would mean something's wrong).

print(data.describe())

# Check for missing values -- markets are closed weekends/holidays, but
# yfinance should already exclude those days, so we expect near-zero NaNs.
print("Missing values per column:")
print(data.isna().sum())

# %% --------------------------------------------------------------------
# STEP 4: Plot the closing price
# --------------------------------------------------------------------
# Always look at your data before modeling it. This single plot will show
# you the dot-com crash recovery, 2008, COVID crash, and recent years.

plt.figure(figsize=(12, 5))
plt.plot(data.index, data["Close"])
plt.title("S&P 500 Closing Price (2000-2026)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(alpha=0.3)
plt.savefig("sp500_price_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# %% --------------------------------------------------------------------
# STEP 5: Compute daily returns -- THIS IS IMPORTANT
# --------------------------------------------------------------------
# Raw price is "non-stationary" (its mean and variance change over time --
# the S&P 500 in 2000 and 2026 are on totally different scales). Almost
# no statistical or ML model works well on non-stationary data directly.
#
# The fix: work with RETURNS instead of prices. Daily return = 
# (today's price - yesterday's price) / yesterday's price
# This is roughly stationary (hovers around a constant mean/variance),
# which is why nearly every finance paper models returns, not prices.

data["Return"] = data["Close"].pct_change()  # pct_change does exactly the formula above

# Drop the first row, which is NaN (no previous day to compare to)
data = data.dropna(subset=["Return"])

print(data[["Close", "Return"]].head())

# %% --------------------------------------------------------------------
# STEP 6: Plot returns -- compare visually to the price plot
# --------------------------------------------------------------------
# Notice: this should look like noise oscillating around zero, with
# periods of big swings (2008, 2020) and periods of calm. That clustering
# of high/low volatility is called "volatility clustering" and is a
# famous stylized fact of financial returns -- it's WHY GARCH models exist.

plt.figure(figsize=(12, 5))
plt.plot(data.index, data["Return"], linewidth=0.5)
plt.title("S&P 500 Daily Returns (2000-2026)")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.grid(alpha=0.3)
plt.savefig("sp500_returns_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# %% --------------------------------------------------------------------
# STEP 7: Distribution of returns
# --------------------------------------------------------------------
# A histogram shows whether returns look "normal" (bell-curve) or not.
# Financial returns are famously "fat-tailed" -- extreme moves happen more
# often than a normal distribution would predict. This matters for which
# classical models are appropriate later.

plt.figure(figsize=(8, 5))
data["Return"].hist(bins=100)
plt.title("Distribution of S&P 500 Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.savefig("sp500_returns_histogram.png", dpi=150, bbox_inches="tight")
plt.show()

print("Skewness:", data["Return"].skew())   # asymmetry: 0 = symmetric
print("Kurtosis:", data["Return"].kurt())   # tail-fatness: 0 = normal-like, higher = fatter tails

# %% --------------------------------------------------------------------
# STEP 8: Save processed data
# --------------------------------------------------------------------
# This goes in data/processed/ -- this is the cleaned version with returns
# computed, ready for feature engineering next.

data.to_csv("sp500_processed.csv")
print("Saved processed data. Day 1 complete.")
