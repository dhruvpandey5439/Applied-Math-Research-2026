# =============================================================================
# DAY 2: Formally test stationarity (ADF test) + check autocorrelation (ACF/PACF)
# =============================================================================
# Run this in the SAME folder as your saved sp500_processed.csv from Day 1.
# You'll need one new library: statsmodels. Install it first:
#
#   python -m pip install statsmodels
#
# GOAL TODAY: confirm with a real statistical test (not just eyeballing plots)
# that price is non-stationary and returns are stationary. Then check whether
# past returns have ANY statistical relationship with future returns -- this
# is the first real clue about how hard your forecasting problem is.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %% --------------------------------------------------------------------
# STEP 1: Load the data you saved yesterday
# --------------------------------------------------------------------
data = pd.read_csv("sp500_processed.csv", index_col=0, parse_dates=True)
print(data.head())

# %% --------------------------------------------------------------------
# STEP 2: Run the ADF test on PRICE
# --------------------------------------------------------------------
# adfuller() runs the test and returns several values. The two we care
# about right now: the test statistic, and the p-value (2nd item returned).
#
# Recall the logic: small p-value (< 0.05) = evidence of "snap-back"
# behavior = stationary. Large p-value = no such evidence = non-stationary.
# We EXPECT price to fail this (large p-value) based on yesterday's plot.

result_price = adfuller(data["Close"])

print("=== ADF Test: PRICE ===")
print(f"Test Statistic: {result_price[0]:.4f}")
print(f"p-value: {result_price[1]:.4f}")
if result_price[1] < 0.05:
    print("-> p < 0.05: evidence of stationarity")
else:
    print("-> p >= 0.05: NO evidence of stationarity (this is what we expect for price)")

# %% --------------------------------------------------------------------
# STEP 3: Run the ADF test on RETURNS
# --------------------------------------------------------------------
# Same test, different column. We EXPECT this one to pass (small p-value).

result_returns = adfuller(data["Return"])

print("\n=== ADF Test: RETURNS ===")
print(f"Test Statistic: {result_returns[0]:.4f}")
print(f"p-value: {result_returns[1]:.6f}")
if result_returns[1] < 0.05:
    print("-> p < 0.05: evidence of stationarity (this is what we expect for returns)")
else:
    print("-> p >= 0.05: NO evidence of stationarity (unexpected -- flag this if it happens)")

# %% --------------------------------------------------------------------
# STEP 4: Autocorrelation -- does yesterday's return relate to today's?
# --------------------------------------------------------------------
# This is the actual heart of your research question. Autocorrelation
# measures: if I know returns from previous days, does that tell me
# ANYTHING about today's return?
#
# ACF (Autocorrelation Function): correlation between the series and
# itself shifted back by k days, for k = 1, 2, 3... up to some max (here 20).
#
# If financial markets were perfectly "efficient" (a real economic theory --
# the Efficient Market Hypothesis), past returns would tell you NOTHING
# about future returns, and ALL these correlations would be statistically
# indistinguishable from zero. If your project finds otherwise, that's a
# genuinely interesting result worth discussing in your paper.
#
# The shaded blue cone in the plot = the "this could just be random noise"
# zone. Bars poking outside the cone = statistically meaningful correlation
# at that lag.

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

plot_acf(data["Return"], lags=20, ax=axes[0])
axes[0].set_title("Autocorrelation (ACF) of S&P 500 Daily Returns")

plot_pacf(data["Return"], lags=20, ax=axes[1])
axes[1].set_title("Partial Autocorrelation (PACF) of S&P 500 Daily Returns")

plt.tight_layout()
plt.savefig("sp500_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.show()

# %% --------------------------------------------------------------------
# STEP 5: Same plots, but on squared returns
# --------------------------------------------------------------------
# Here's something subtle and important: returns themselves might show
# almost no autocorrelation (hard to predict DIRECTION from past returns),
# but the SIZE of returns (volatility) often does show strong
# autocorrelation -- big moves cluster together (remember "volatility
# clustering" from Day 1?). Squaring the returns removes the sign (+/-)
# and leaves only magnitude, which lets us check this directly.
#
# This is WHY GARCH models exist -- they're specifically built to predict
# volatility (magnitude of moves), not direction, using exactly this
# clustering pattern.

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

plot_acf(data["Return"]**2, lags=20, ax=axes[0])
axes[0].set_title("Autocorrelation of SQUARED Returns (volatility clustering check)")

plot_pacf(data["Return"]**2, lags=20, ax=axes[1])
axes[1].set_title("Partial Autocorrelation of SQUARED Returns")

plt.tight_layout()
plt.savefig("sp500_acf_pacf_squared.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDay 2 complete.")