# PART 12: LSTM (LONG SHORT-TERM MEMORY)
# What LSTM is:
# LSTM is a type of Recurrent Neural Network (RNN) designed to learn
# patterns from SEQUENCES of data. Every other model in this project
# treats each part independently -- it gets today's features and predicts
# tomorrow. LSTM is different: it reads a WINDOW of past days in order
# and maintains a memory of what it has seen.
#
# Why this matters for finance:
# Markets have regime-dependent behavior -- patterns that span many days
# or weeks, not just yesterday. LSTM is theoretically capable of learning
# these longer-range dependencies. Whether it actually DOES on noisy
# financial data is exactly what we will test.
#
# The three gates (simplified):
#   Forget gate:  decides what to erase from memory
#   Input gate:   decides what new information to store
#   Output gate:  decides what to output from memory
#
# Each gate is a small neural network inside the LSTM cell.
# Together they let the model selectively remember and forget
# across sequences of arbitrary length.
#
# How this fits the project:
# LSTM is the peak of the ML complexity ladder in this study.
# The research question is answered by comparing LSTM's accuracy
# against ARIMA, GARCH, GBM, Logistic, RF, and XGBoost.
# More complexity should mean more accuracy -- but does it?


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"]  = "#0e0e0e"
plt.rcParams["axes.facecolor"]    = "#1a1a1a"
plt.rcParams["savefig.facecolor"] = "#0e0e0e"


# STEP 1: LOAD DATA


script_dir = os.path.dirname(os.path.abspath(__file__))

features_path = os.path.join(
    script_dir, "..", "data processed", "sp500_features.csv"
)

data = pd.read_csv(features_path, index_col=0, parse_dates=True)
data = data.dropna()

print(f"Data loaded: {data.index[0].date()} to {data.index[-1].date()} "
      f"({len(data)} rows)")

feature_cols = [
    "return_lag_1", "return_lag_2", "return_lag_3",
    "return_lag_5", "return_ma_5",  "return_ma_10",
    "volatility_5", "volatility_10", "volatility_20"
]

X_raw = data[feature_cols].values
y     = data["target_direction"].values

print(f"Features: {feature_cols}")
print(f"Total samples: {len(X_raw)}\n")

# STEP 2: SCALE FEATURES

# LSTM is a neural network -- it is sensitive to feature scale.
# We scale the ENTIRE feature matrix here using StandardScaler.
#
# IMPORTANT NOTE ON SCALING FOR LSTM:
# Ideally we would scale inside each fold to prevent any leakage.
# In practice for financial returns -- which are already small,
# mean-zero, and similarly scaled -- global scaling introduces
# negligible leakage. We note this as a minor limitation.
#
# StandardScaler: transforms each feature to mean=0, std=1.
# This ensures no single feature dominates due to scale alone.


scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("Features scaled with StandardScaler (global fit -- see comment above).")

# STEP 3: CREATE SEQUENCES

# LSTM does not take single-day feature vectors as input.
# It takes sequences/windows of consecutive days.
#
# LOOKBACK = 20 means: for each prediction, give the model the
# past 20 days of features as a 3D input:
#   (samples, timesteps, features) = (N, 20, 9)
#
# Example:
#   To predict day 21's direction, input = days 1-20 features
#   To predict day 22's direction, input = days 2-21 features
#   And so on, sliding the window forward one day at a time.
#
# Why 20 days?
# 20 trading days ≈ 1 calendar month. A reasonable window to
# capture short-to-medium term patterns without excessive memory.


LOOKBACK = 20

def create_sequences(X, y, lookback):
    """
    Converts flat feature array into overlapping sequences for LSTM.

    Parameters:
        X        : scaled feature array (n_samples, n_features)
        y        : target array (n_samples,)
        lookback : number of past days in each sequence

    Returns:
        X_seq : array of shape (n_samples - lookback, lookback, n_features)
        y_seq : array of shape (n_samples - lookback,)

    Note:
        The first `lookback` samples are sacrificed -- we cannot build
        a full sequence for them since they don't have enough history.
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y, LOOKBACK)

print(f"Sequences created:")
print(f"  X_seq shape: {X_seq.shape}  (samples, timesteps, features)")
print(f"  y_seq shape: {y_seq.shape}\n")


# STEP 4: WALK-FORWARD VALIDATION SETUP

# Same walk-forward logic as all previous parts.
# We use 3 folds instead of 5 because LSTM retrains a full neural
# network each fold -- 5 folds would take too long.
#
# Each fold: train on earlier sequences, validate on later sequences.
# The test window never overlaps with training -- strictly chronological.


n        = len(X_seq)
n_splits = 3
test_size  = int(n * 0.12)
min_train  = int(n * 0.50)

folds = []
for i in range(n_splits):
    test_end   = n - (n_splits - 1 - i) * test_size
    test_start = test_end - test_size
    train_end  = test_start
    if train_end < min_train:
        continue
    folds.append((0, train_end, test_start, test_end))

print(f"Walk-forward validation: {len(folds)} folds")
for i, (_, train_end, test_start, test_end) in enumerate(folds):
    print(f"  Fold {i+1}: Train 0→{train_end} | Val {test_start}→{test_end} "
          f"({test_end - test_start} samples)")
print()


# STEP 5: BUILD LSTM MODEL

# Architecture:
#   Layer 1: LSTM(64 units) -- reads the 20-day sequence, returns full sequence
#   Dropout(0.2)             -- randomly zeros 20% of connections each batch
#   Layer 2: LSTM(32 units) -- reads output of layer 1, returns final state only
#   Dropout(0.2)             -- same dropout for regularization
#   Dense(1, sigmoid)        -- outputs probability of "up" (0 to 1)
#
# Why two LSTM layers?
# The first layer learns low-level temporal patterns (e.g. short momentum).
# The second layer learns higher-level patterns from those (e.g. regime shifts).
# Stacking gives more power than a single layer.
#
# Why Dropout?
# Neural networks with many parameters easily memorize training data.
# Dropout randomly disables neurons during training, forcing the network
# to learn redundant representations. Reduces overfitting.
#
# Why sigmoid output?
# We need a probability between 0 and 1 for binary classification.
# Sigmoid maps any real number to (0,1).
# Output > 0.5 → predict "up". Output <= 0.5 → predict "down".


def build_lstm(input_shape):
    """
    Builds and compiles the LSTM model.

    Parameters:
        input_shape : tuple (lookback, n_features) = (20, 9)

    Returns:
        compiled Keras Sequential model
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    # Binary crossentropy: standard loss for binary classification.
    # Measures how far predicted probabilities are from true labels.
    # Adam: adaptive learning rate optimizer. Adjusts step size
    # automatically during training. Standard choice for deep learning.

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Print model summary once before training
sample_model = build_lstm((LOOKBACK, len(feature_cols)))
print("LSTM Architecture:")
sample_model.summary()
print()


# STEP 6: WALK-FORWARD TRAINING LOOP


fold_results   = []
all_train_loss = []
all_val_loss   = []

for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):

    print(f"{'='*50}")
    print(f"Fold {fold_idx + 1} / {len(folds)}")
    print(f"  Train samples: {train_end - train_start}")
    print(f"  Val samples:   {test_end - test_start}")

    X_train = X_seq[train_start:train_end]
    y_train = y_seq[train_start:train_end]
    X_val   = X_seq[test_start:test_end]
    y_val   = y_seq[test_start:test_end]

    # Build a fresh model each fold -- no weight carryover between folds
    model = build_lstm((LOOKBACK, len(feature_cols)))

    # EarlyStopping: monitors validation loss.
    # If val_loss does not improve for 15 consecutive epochs, stop training.
    # restore_best_weights=True: revert to the epoch with lowest val_loss.
    # This prevents training too long and overfitting.
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0
    )

    epochs_run = len(history.history["loss"])
    print(f"  Epochs run: {epochs_run} (stopped early at patience=15)")

    # Generate predictions
    preds_prob = model.predict(X_val, verbose=0).flatten()
    preds      = (preds_prob > 0.5).astype(int)

    acc = accuracy_score(y_val, preds)
    f1  = f1_score(y_val, preds, zero_division=0)

    print(f"  Direction Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 Score:           {f1:.4f}")

    fold_results.append({
        "fold":        fold_idx + 1,
        "accuracy":    acc,
        "f1_score":    f1,
        "epochs_run":  epochs_run,
        "train_loss":  history.history["loss"],
        "val_loss":    history.history["val_loss"],
    })

    all_train_loss.append(history.history["loss"])
    all_val_loss.append(history.history["val_loss"])

print(f"{'='*50}\n")


# STEP 7: AGGREGATE RESULTS


results_df = pd.DataFrame([{
    "fold":       r["fold"],
    "accuracy":   r["accuracy"],
    "f1_score":   r["f1_score"],
    "epochs_run": r["epochs_run"],
} for r in fold_results])

mean_acc = results_df["accuracy"].mean()
mean_f1  = results_df["f1_score"].mean()
naive_baseline = 0.5393

print("=" * 60)
print("LSTM SUMMARY ACROSS ALL FOLDS")
print("=" * 60)
print(f"Mean Direction Accuracy:  {mean_acc:.4f} ({mean_acc*100:.2f}%)")
print(f"Mean F1 Score:            {mean_f1:.4f}")
print(f"vs Naive Baseline:        {(mean_acc - naive_baseline)*100:+.2f} pp")
print(f"Mean Epochs Run:          {results_df['epochs_run'].mean():.1f}")
print("=" * 60)


# STEP 8: SAVE RESULTS


results_dir     = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)
comparison_path = os.path.join(results_dir, "model_comparison.csv")

new_row = pd.DataFrame([{
    "model":    "LSTM",
    "accuracy": round(mean_acc, 4),
    "f1_score": round(mean_f1, 4),
    "rmse":     None,
    "mae":      None,
    "test_start": "walk-forward (3 folds)",
    "test_end":   "walk-forward (3 folds)",
    "notes": (
        f"Deep learning. 2-layer LSTM (64→32 units). Lookback=20 days. "
        f"Dropout=0.2. EarlyStopping patience=15. "
        f"Mean epochs: {results_df['epochs_run'].mean():.1f}. "
        f"vs naive baseline: {(mean_acc - naive_baseline)*100:+.2f}pp."
    ),
}])

if os.path.exists(comparison_path):
    existing = pd.read_csv(comparison_path)
    existing = existing[existing["model"] != "LSTM"]
    combined = pd.concat([existing, new_row], ignore_index=True)
else:
    combined = new_row

combined.to_csv(comparison_path, index=False)
print(f"\nSaved results to: {comparison_path}")


# STEP 9: VISUALIZATIONS


figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Part 12: LSTM Results", fontsize=14,
             fontweight="bold", color="white")

# --- Plot 1: Training vs Validation Loss (last fold) ---
ax1         = axes[0]
last_train  = all_train_loss[-1]
last_val    = all_val_loss[-1]
epochs_axis = range(1, len(last_train) + 1)
ax1.plot(epochs_axis, last_train, color="#5cb8e0", linewidth=1.5,
         label="Train loss")
ax1.plot(epochs_axis, last_val,   color="#e05c5c", linewidth=1.5,
         label="Val loss")
ax1.set_xlabel("Epoch", color="white")
ax1.set_ylabel("Loss", color="white")
ax1.set_title("Training vs Validation Loss\n(Last Fold)", color="white")
ax1.legend(fontsize=9)
ax1.tick_params(colors="white")

# --- Plot 2: Direction Accuracy per Fold ---
ax2       = axes[1]
fold_nums = results_df["fold"].tolist()
accs      = results_df["accuracy"].tolist()
colors = ["#e05c5c" if a < naive_baseline else "#5ce0a0" for a in accs]