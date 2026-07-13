import os
import pandas as pd


# STEP 1: Load Day 3's feature dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sp500_features.csv")
data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

print("Loaded data shape:", data.shape)
print("Date range:", data.index.min(), "to", data.index.max())


# STEP 2: The WRONG way -- random shuffling (we do this ONLY to show why
# it's wrong, never actually use train_test_split like this for real)

from sklearn.model_selection import train_test_split
wrong_train, wrong_test = train_test_split(data, test_size=0.2, shuffle=True)

print("\n=== WRONG approach (randomly shuffled) ===")
print("Train date range:", wrong_train.index.min(), "to", wrong_train.index.max())
print("Test date range:", wrong_test.index.min(), "to", wrong_test.index.max())
# The "training" set contains rows from 2026, and the "test" set contains rows
# from 2001. A model trained this way has already "seen" the future
# relative to many of its own test points -- the evaluation is meaningless.


# STEP 3: The RIGHT way -- a single chronological split

# Pick a cutoff point. Everything before it = train. Everything after it =
# test. No shuffling, no mixing -- this respects time order completely,
# the same principle as shift(1) only ever looking backward.

split_index = int(len(data) * 0.8)  # 80% of rows, in original order
train = data.iloc[:split_index]
test = data.iloc[split_index:]

print("\n=== RIGHT approach (chronological split) ===")
print("Train date range:", train.index.min(), "to", train.index.max())
print("Test date range:", test.index.min(), "to", test.index.max())
print("Train size:", train.shape[0], "rows")
print("Test size:", test.shape[0], "rows")
# Now: train is entirely EARLIER than test. No overlap, no future leakage.


# STEP 4: Walk-forward validation -- a more rigorous version

# A single train/test split gives you exactly ONE number to judge a model
# by -- risky, because that one test period might happen to be unusually
# easy or hard (e.g. testing only on 2020's COVID crash would be misleading
# either way). Walk-forward validation repeats the chronological split
# multiple times with an EXPANDING training window, giving you many test
# results instead of one.
#
# Visualize it like this, with 5 folds:
#   Fold 1: train [2000-2015] -> test [2016]
#   Fold 2: train [2000-2016] -> test [2017]
#   Fold 3: train [2000-2017] -> test [2018]
#   Fold 4: train [2000-2018] -> test [2019]
#   Fold 5: train [2000-2019] -> test [2020]
#
# Training data always grows (more history = usually better), but test
# data is always strictly AFTER its corresponding training data. This is
# what every model from here on will actually be evaluated with.

from sklearn.model_selection import TimeSeriesSplit

n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

print(f"\n=== Walk-forward validation: {n_splits} folds ===")
for fold, (train_idx, test_idx) in enumerate(tscv.split(data), start=1):
    fold_train = data.iloc[train_idx]
    fold_test = data.iloc[test_idx]
    print(f"\nFold {fold}:")
    print(f"  Train: {fold_train.index.min().date()} to {fold_train.index.max().date()} ({len(fold_train)} rows)")
    print(f"  Test:  {fold_test.index.min().date()} to {fold_test.index.max().date()} ({len(fold_test)} rows)")
    # Sanity check: every fold's test data should start the day right after
    # that fold's training data ends.


# STEP 5: Save the simple chronological split for immediate use

# We'll use TimeSeriesSplit (Step 4) directly inside model training scripts
# going forward, but saving this simple 80/20 version now gives us a quick
# train/test pair to sanity-check the very first models with.
plt.style.use("dark_background")
train_path = os.path.join(script_dir, "sp500_train.csv")
test_path = os.path.join(script_dir, "sp500_test.csv")
train.to_csv(train_path)
test.to_csv(test_path)
print(f"\nSaved train set ({train.shape[0]} rows) and test set ({test.shape[0]} rows).")
print("Day 4 complete.")