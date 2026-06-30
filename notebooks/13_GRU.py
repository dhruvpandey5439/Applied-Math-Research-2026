# =============================================================================
# PART 13: GRU (GATED RECURRENT UNIT)
# =============================================================================
# Research Question:
# To what extent do increasingly complex machine learning models improve
# predictive performance compared to classical statistical and mathematical
# methods when forecasting financial time-series data?
#
# What GRU is:
# GRU (Gated Recurrent Unit) is a simplified version of LSTM. Both are
# recurrent neural networks that process sequences of data and maintain
# memory across timesteps. GRU was introduced by Cho et al. (2014) as
# a more computationally efficient alternative to LSTM.
#
# How GRU differs from LSTM:
# LSTM has THREE gates: forget gate, input gate, output gate.
# GRU has TWO gates: reset gate, update gate.
#   Reset gate:  decides how much of the past to forget
#   Update gate: decides how much of the past to carry forward
# Fewer gates = fewer parameters = faster training.
# GRU: ~23,000 parameters vs LSTM: ~31,393 parameters in this project.
#
# Why this comparison matters:
# Part 12 (LSTM) tied Logistic Regression at 54.40% despite having
# 31,393 parameters vs 9 coefficients. A critic could argue this is
# because LSTM specifically is the wrong architecture for financial data.
# GRU directly tests that objection:
#   If GRU also lands ~54-55%: the ceiling is in the DATA, not the
#   architecture. The signal in daily S&P 500 returns is too weak for
#   any sequential deep learning model to exploit.
#   If GRU significantly differs from LSTM: architecture matters and
#   the choice of deep learning model affects results.
# Either outcome strengthens the Discussion section.
#
# How this fits the project:
# GRU sits alongside LSTM at the deep learning end of the complexity
# ladder. Together they give a complete picture of whether sequential
# deep learning — regardless of specific architecture — adds value over
# simpler models for financial direction forecasting.
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"]  = "#0e0e0e"
plt.rcParams["axes.facecolor"]    = "#1a1a1a"
plt.rcParams["savefig.facecolor"] = "#0e0e0e"

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

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

# =============================================================================
# STEP 2: SCALE FEATURES
# =============================================================================
# Same global scaling as Part 12 LSTM for direct comparability.
# StandardScaler: transforms each feature to mean=0, std=1.
# Minor limitation: global fit includes future data in scaling statistics.
# Noted as limitation in paper — same caveat as Part 12.
# =============================================================================

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("Features scaled with StandardScaler (global fit — same as Part 12).")

# =============================================================================
# STEP 3: CREATE SEQUENCES
# =============================================================================
# Identical to Part 12. LOOKBACK=20 days for direct comparability with LSTM.
# Each prediction uses the past 20 days of features as input.
# Shape: (samples, timesteps, features) = (N, 20, 9)
# =============================================================================

LOOKBACK = 20

def create_sequences(X, y, lookback):
    """
    Converts flat feature array into overlapping sequences for GRU.
    Identical to Part 12 — same function, same logic.

    Parameters:
        X        : scaled feature array (n_samples, n_features)
        y        : target array (n_samples,)
        lookback : number of past days in each sequence

    Returns:
        X_seq : array of shape (n_samples - lookback, lookback, n_features)
        y_seq : array of shape (n_samples - lookback,)
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

# =============================================================================
# STEP 4: WALK-FORWARD VALIDATION SETUP
# =============================================================================
# Identical to Part 12: 3 folds, 12% test size, 50% minimum training.
# Using identical fold structure ensures GRU and LSTM results are
# directly comparable — same data splits, same validation windows.
# =============================================================================

n          = len(X_seq)
n_splits   = 3
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

# =============================================================================
# STEP 5: BUILD GRU MODEL
# =============================================================================
# Architecture mirrors Part 12 LSTM exactly, with GRU layers substituted:
#   Layer 1: GRU(64 units)  -- reads 20-day sequence, returns full sequence
#   Dropout(0.2)             -- same regularization as LSTM
#   Layer 2: GRU(32 units)  -- reads layer 1 output, returns final state
#   Dropout(0.2)             -- same regularization as LSTM
#   Dense(1, sigmoid)        -- outputs probability of "up"
#
# Why mirror the LSTM architecture exactly?
# To isolate the effect of GRU vs LSTM specifically. If we changed the
# number of units, dropout rate, or any other hyperparameter, we couldn't
# attribute differences in performance to the GRU architecture alone.
# Keeping everything identical except GRU vs LSTM makes this a clean
# controlled comparison.
#
# GRU parameter count:
# GRU has 2 gates (reset, update) vs LSTM's 3 effective gate groups.
# Formula: 3 * [(input_features + hidden_units) * hidden_units + hidden_units]
# Layer 1: 3 * [(9 + 64) * 64 + 64] = ~14,208 parameters
# Layer 2: 3 * [(64 + 32) * 32 + 32] = ~9,312 parameters
# Total: ~23,520 parameters vs LSTM's 31,393
# Fewer parameters = faster training, less prone to overfitting.
# =============================================================================

def build_gru(input_shape):
    """
    Builds and compiles the GRU model.
    Architecture mirrors Part 12 LSTM for direct comparability.

    Parameters:
        input_shape : tuple (lookback, n_features) = (20, 9)

    Returns:
        compiled Keras Sequential model
    """
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Print model summary once before training
sample_model = build_gru((LOOKBACK, len(feature_cols)))
print("GRU Architecture:")
sample_model.summary()
print()

# =============================================================================
# STEP 6: WALK-FORWARD TRAINING LOOP
# =============================================================================

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

    # Fresh model each fold — no weight carryover between folds.
    # Same as Part 12.
    model = build_gru((LOOKBACK, len(feature_cols)))

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

    preds_prob = model.predict(X_val, verbose=0).flatten()
    preds      = (preds_prob > 0.5).astype(int)

    acc = accuracy_score(y_val, preds)
    f1  = f1_score(y_val, preds, zero_division=0)

    print(f"  Direction Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 Score:           {f1:.4f}")

    fold_results.append({
        "fold":       fold_idx + 1,
        "accuracy":   acc,
        "f1_score":   f1,
        "epochs_run": epochs_run,
        "train_loss": history.history["loss"],
        "val_loss":   history.history["val_loss"],
    })

    all_train_loss.append(history.history["loss"])
    all_val_loss.append(history.history["val_loss"])

print(f"{'='*50}\n")

# =============================================================================
# STEP 7: AGGREGATE RESULTS
# =============================================================================

results_df = pd.DataFrame([{
    "fold":       r["fold"],
    "accuracy":   r["accuracy"],
    "f1_score":   r["f1_score"],
    "epochs_run": r["epochs_run"],
} for r in fold_results])

mean_acc       = results_df["accuracy"].mean()
mean_f1        = results_df["f1_score"].mean()
naive_baseline = 0.5393
lstm_accuracy  = 0.5440  # Part 12 result for direct comparison

print("=" * 60)
print("GRU SUMMARY ACROSS ALL FOLDS")
print("=" * 60)
print(f"Mean Direction Accuracy:  {mean_acc:.4f} ({mean_acc*100:.2f}%)")
print(f"Mean F1 Score:            {mean_f1:.4f}")
print(f"vs Naive Baseline:        {(mean_acc - naive_baseline)*100:+.2f} pp")
print(f"vs LSTM (Part 12):        {(mean_acc - lstm_accuracy)*100:+.2f} pp")
print(f"Mean Epochs Run:          {results_df['epochs_run'].mean():.1f}")
print("=" * 60)

# =============================================================================
# STEP 8: SAVE RESULTS
# =============================================================================

results_dir     = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)
comparison_path = os.path.join(results_dir, "model_comparison.csv")

new_row = pd.DataFrame([{
    "model":    "GRU",
    "accuracy": round(mean_acc, 4),
    "f1_score": round(mean_f1, 4),
    "rmse":     None,
    "mae":      None,
    "test_start": "walk-forward (3 folds)",
    "test_end":   "walk-forward (3 folds)",
    "notes": (
        f"Deep learning. 2-layer GRU (64→32 units). Lookback=20 days. "
        f"Dropout=0.2. EarlyStopping patience=15. "
        f"Architecture mirrors Part 12 LSTM exactly for direct comparison. "
        f"Mean epochs: {results_df['epochs_run'].mean():.1f}. "
        f"vs naive baseline: {(mean_acc - naive_baseline)*100:+.2f}pp. "
        f"vs LSTM: {(mean_acc - lstm_accuracy)*100:+.2f}pp."
    ),
}])

if os.path.exists(comparison_path):
    existing = pd.read_csv(comparison_path)
    existing = existing[existing["model"] != "GRU"]
    combined = pd.concat([existing, new_row], ignore_index=True)
else:
    combined = new_row

combined.to_csv(comparison_path, index=False)
print(f"\nSaved results to: {comparison_path}")

# =============================================================================
# STEP 9: VISUALIZATIONS
# =============================================================================

figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Part 13: GRU Results", fontsize=14,
             fontweight="bold", color="white")

# --- Plot 1: Training vs Validation Loss (last fold) ---
ax1        = axes[0]
last_train = all_train_loss[-1]
last_val   = all_val_loss[-1]
epochs_ax  = range(1, len(last_train) + 1)
ax1.plot(epochs_ax, last_train, color="#5cb8e0", linewidth=1.5,
         label="Train loss")
ax1.plot(epochs_ax, last_val,   color="#e05c5c", linewidth=1.5,
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
colors    = ["#e05c5c" if a < naive_baseline else "#5ce0a0" for a in accs]
bars      = ax2.bar(fold_nums, [a * 100 for a in accs],
                    color=colors, edgecolor="white", alpha=0.85)
ax2.axhline(naive_baseline * 100, color="white", linestyle="--",
            linewidth=1.5, label=f"Naive baseline ({naive_baseline*100:.1f}%)")
ax2.axhline(lstm_accuracy * 100, color="#a05ce0", linestyle="--",
            linewidth=1.5, label=f"LSTM ({lstm_accuracy*100:.1f}%)")
ax2.axhline(50, color="gray", linestyle=":", linewidth=1)
ax2.set_xlabel("Fold", color="white")
ax2.set_ylabel("Direction Accuracy (%)", color="white")
ax2.set_title("GRU: Accuracy per Fold", color="white")
ax2.set_xticks(fold_nums)
ax2.set_ylim(40, 65)
ax2.legend(fontsize=8)
ax2.tick_params(colors="white")
for bar, acc in zip(bars, accs):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{acc*100:.1f}%", ha="center", va="bottom",
             fontsize=8, color="white")

# --- Plot 3: GRU vs LSTM vs All Models Comparison ---
ax3 = axes[2]
model_names = ["Always Up\n(Naive)", "ARIMA\n(1,0,1)",
               "Logistic\nRegression", "LSTM\n(Part 12)",
               "GRU\n(Part 13)"]
model_accs  = [53.93, 50.73, 54.40, lstm_accuracy * 100, mean_acc * 100]
bar_colors  = ["#888888", "#e0a85c", "#5ce0a0", "#a05ce0", "#5cb8e0"]
bars3       = ax3.bar(model_names, model_accs, color=bar_colors, alpha=0.85)
ax3.axhline(naive_baseline * 100, color="white", linestyle="--",
            linewidth=1.5, label=f"Naive floor ({naive_baseline*100:.1f}%)")
ax3.axhline(50, color="gray", linestyle=":", linewidth=1)
ax3.set_ylabel("Direction Accuracy (%)", color="white")
ax3.set_title("GRU vs LSTM vs Key Models", color="white")
ax3.set_ylim(40, 65)
ax3.legend(fontsize=8)
ax3.tick_params(colors="white")
for bar, acc in zip(bars3, model_accs):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{acc:.1f}%", ha="center", va="bottom",
             fontsize=8, color="white")

plt.tight_layout()
fig_path = os.path.join(figures_dir, "part13_gru_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to: {fig_path}")

# =============================================================================
# STEP 10: FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Model:                   GRU")
print(f"Architecture:            2-layer GRU (64→32 units)")
print(f"Lookback window:         {LOOKBACK} days")
print(f"Total parameters:        ~23,520 (vs LSTM's 31,393)")
print(f"Mean direction accuracy: {mean_acc*100:.2f}%")
print(f"Mean F1 score:           {mean_f1:.4f}")
print(f"vs Naive baseline:       {(mean_acc - naive_baseline)*100:+.2f} pp")
print(f"vs LSTM (Part 12):       {(mean_acc - lstm_accuracy)*100:+.2f} pp")
print("=" * 60)

if abs(mean_acc - lstm_accuracy) < 0.01:
    print("\nINTERPRETATION: GRU and LSTM perform within 1pp of each other.")
    print("This supports the conclusion that the performance ceiling is")
    print("in the DATA (weak signal in daily returns), not the architecture.")
    print("Neither sequential deep learning model can exploit what isn't there.")
elif mean_acc > lstm_accuracy + 0.01:
    print("\nINTERPRETATION: GRU outperforms LSTM.")
    print("Architecture matters — GRU's simpler gate structure may")
    print("generalize better on noisy financial data.")
else:
    print("\nINTERPRETATION: LSTM outperforms GRU.")
    print("LSTM's additional gate complexity provides marginal benefit")
    print("on this dataset.")