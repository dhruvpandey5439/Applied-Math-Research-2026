
import os
import pandas as pd
import numpy as np
 
# %% --------------------------------------------------------------------
# STEP 1: Load Day 1's processed data
# --------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sp500_processed.csv")
data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
 
print("Starting shape:", data.shape)
print(data[["Close", "Return"]].head())
 
# %% --------------------------------------------------------------------
# STEP 2: Lagged returns -- "what was the return N days ago?"
# --------------------------------------------------------------------
# .shift(1) moves every value down by 1 row. So Return.shift(1) on a given
# row = the return from the PREVIOUS row, i.e. yesterday. This is exactly
# how you avoid lookahead: shift(N) always pulls from N days in the PAST.
 
for lag in [1, 2, 3, 5]:
    data[f"return_lag_{lag}"] = data["Return"].shift(lag)
 
print("\nLagged return columns (first 10 rows):")
print(data[["Return", "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5"]].head(10))

# %% --------------------------------------------------------------------
# STEP 3: Rolling (moving) average of returns -- smooths out daily noise
# --------------------------------------------------------------------
# .rolling(window=N) looks at the last N rows INCLUDING the current one, so
# we add .shift(1) afterward to push it back one day -- otherwise today's
# return would leak into "today's feature", which is lookahead bias.
#
# A 5-day rolling average answers: "on average, how have returns looked over
# about the last trading week?" Smooths out single noisy days.
 
data["return_ma_5"] = data["Return"].rolling(window=5).mean().shift(1)
data["return_ma_10"] = data["Return"].rolling(window=10).mean().shift(1)
 
print("\nRolling average columns (rows 10-15):")
print(data[["Return", "return_ma_5", "return_ma_10"]].iloc[10:15])
 
# %% --------------------------------------------------------------------
# STEP 4: Rolling volatility -- THIS IS THE IMPORTANT ONE
# --------------------------------------------------------------------
# Standard deviation measures how spread out / how "jumpy" a set of numbers
# is. Rolling standard deviation of returns over the last N days = a direct
# measure of recent volatility. Given yesterday's finding (volatility
# clusters strongly, out to 20+ days), this is expected to be your most
# informative feature.
 
data["volatility_5"] = data["Return"].rolling(window=5).std().shift(1)
data["volatility_10"] = data["Return"].rolling(window=10).std().shift(1)
data["volatility_20"] = data["Return"].rolling(window=20).std().shift(1)
 
print("\nRolling volatility columns (rows 25-30):")
print(data[["Return", "volatility_5", "volatility_10", "volatility_20"]].iloc[25:30])
 
# %% --------------------------------------------------------------------
# STEP 5: Define the TARGETS -- what we're actually trying to predict
# --------------------------------------------------------------------
# target_direction: 1 if tomorrow's return is positive, 0 if negative/flat.
#   This uses Return.shift(-1) -- shifting BACKWARD in time -- which pulls
#   TOMORROW's return into today's row. This is the one and only place
#   "looking forward" is correct, because it's literally defining the
#   answer we want to predict, not a feature we feed the model as input.
#
# target_return: tomorrow's actual return value (for regression models).
 
data["target_direction"] = (data["Return"].shift(-1) > 0).astype(int)
data["target_return"] = data["Return"].shift(-1)
 
print("\nTargets (first 10 rows):")
print(data[["Return", "target_direction", "target_return"]].head(10))
# Sanity check: target_return on row N should exactly equal Return on row N+1.
# target_direction on row N should be 1 exactly when target_return on row N is positive.
 
# %% --------------------------------------------------------------------
# STEP 6: Drop rows with NaN (from the lags/rolling windows at the start,
# and from the shift(-1) target at the very end)
# --------------------------------------------------------------------
before = data.shape[0]
data = data.dropna()
after = data.shape[0]
print(f"\nDropped {before - after} rows with missing values (expected from lag/rolling windows).")
print("Final shape:", data.shape)
 
# %% --------------------------------------------------------------------
# STEP 7: Quick sanity check -- class balance of target_direction
# --------------------------------------------------------------------
# If one class (up days) vastly outnumbers the other (down days), that
# affects how we judge "accuracy" later -- a model could get high accuracy
# just by always predicting "up" without learning anything real.
 
print("\nDirection target balance:")
print(data["target_direction"].value_counts())
print(data["target_direction"].value_counts(normalize=True))
plt.style.use("dark_background")

# %% --------------------------------------------------------------------
# STEP 8: Save the feature-engineered dataset
# --------------------------------------------------------------------
output_path = os.path.join(script_dir, "sp500_features.csv")
data.to_csv(output_path)
print(f"\nSaved feature dataset to {output_path}")
print("Day 3 complete.")