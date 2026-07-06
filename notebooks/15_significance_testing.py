# =============================================================================
# PART 15: SIGNIFICANCE TESTING
# =============================================================================
# Research Question:
# To what extent do increasingly complex machine learning models improve
# predictive performance compared to classical statistical and mathematical
# methods when forecasting financial time-series data?
#
# What this part does:
# Tests whether the accuracy differences between models are statistically
# significant or just noise. Two tests are used:
#
# 1. Diebold-Mariano (DM) test — for key S&P 500 model pairs
#    Requires raw prediction arrays. We re-run 4 key models with
#    prediction saving enabled, then compare forecast errors directly.
#    This is the gold standard for forecast comparison in financial
#    econometrics literature.
#
# 2. Binomial significance test — for all models and all assets
#    Only requires accuracy + sample size. Tests whether a model's
#    accuracy is significantly above the naive baseline.
#    Covers all assets without requiring re-running the full pipeline.
#
# Why significance testing matters:
# A 0.47pp accuracy difference could be real signal or random noise.
# Without a significance test you cannot claim one model is genuinely
# better than another. p < 0.05 means the difference is statistically
# significant. p > 0.05 means it could be noise — which is itself
# a valid and honest finding.
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"]  = "#0e0e0e"
plt.rcParams["axes.facecolor"]    = "#1a1a1a"
plt.rcParams["savefig.facecolor"] = "#0e0e0e"

script_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# SECTION 1: RE-RUN KEY S&P 500 MODELS WITH PREDICTION SAVING
# =============================================================================
# We re-run 4 models on S&P 500 with prediction arrays saved to disk.
# Models: Naive Baseline, ARIMA, Logistic Regression, LSTM
# These are the 4 most important comparisons for the DM test.
# =============================================================================

print("=" * 60)
print("SECTION 1: RE-RUNNING KEY S&P 500 MODELS")
print("=" * 60)

# --- Load S&P 500 features ---
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

returns  = data["Return"].values
y_dir    = data["target_direction"].values
X_raw    = data[feature_cols].values
n        = len(data)

# Walk-forward fold setup — same as all previous parts
def get_folds(n, n_splits=3, test_frac=0.12, min_train_frac=0.50):
    test_size = int(n * test_frac)
    min_train = int(n * min_train_frac)
    folds = []
    for i in range(n_splits):
        test_end   = n - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        train_end  = test_start
        if train_end < min_train:
            continue
        folds.append((0, train_end, test_start, test_end))
    return folds

folds = get_folds(n, n_splits=3)

# Storage for predictions and true labels
saved_preds = {}
saved_true  = {}

# =============================================================================
# MODEL 1: NAIVE BASELINE — save predictions
# =============================================================================
print("\n--- Re-running: Naive Baseline ---")

naive_preds = []
naive_true  = []

for (_, train_end, test_start, test_end) in folds:
    y_val = y_dir[test_start:test_end]
    preds = np.ones(len(y_val), dtype=int)
    naive_preds.extend(preds)
    naive_true.extend(y_val)

naive_preds = np.array(naive_preds)
naive_true  = np.array(naive_true)

saved_preds["naive"] = naive_preds
saved_true["naive"]  = naive_true

naive_acc = np.mean(naive_preds == naive_true)
print(f"Naive Baseline Accuracy: {naive_acc*100:.2f}%")

# =============================================================================
# MODEL 2: ARIMA — save predictions
# =============================================================================
print("\n--- Re-running: ARIMA(1,0,1) ---")
print("(This will take several minutes...)")

from statsmodels.tsa.arima.model import ARIMA as ARIMAModel

arima_preds = []
arima_true  = []

for fold_idx, (_, train_end, test_start, test_end) in enumerate(folds):
    fold_train = returns[:train_end]
    fold_val   = returns[test_start:test_end]
    fold_true  = y_dir[test_start:test_end]

    fold_preds = []
    for i in range(len(fold_val)):
        history = list(fold_train) + list(fold_val[:i])
        try:
            model = ARIMAModel(history, order=(1, 0, 1))
            fit   = model.fit()
            forecast = fit.forecast(steps=1)[0]
        except Exception:
            forecast = 0.0
        fold_preds.append(1 if forecast > 0 else 0)

    arima_preds.extend(fold_preds)
    arima_true.extend(fold_true)
    acc = np.mean(np.array(fold_preds) == fold_true)
    print(f"  Fold {fold_idx+1}: {acc*100:.2f}%")

arima_preds = np.array(arima_preds)
arima_true  = np.array(arima_true)

saved_preds["arima"] = arima_preds
saved_true["arima"]  = arima_true

arima_acc = np.mean(arima_preds == arima_true)
print(f"ARIMA Accuracy: {arima_acc*100:.2f}%")

# =============================================================================
# MODEL 3: LOGISTIC REGRESSION — save predictions
# =============================================================================
print("\n--- Re-running: Logistic Regression ---")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

log_preds = []
log_true  = []

for fold_idx, (_, train_end, test_start, test_end) in enumerate(folds):
    X_tr  = X_raw[:train_end]
    X_val = X_raw[test_start:test_end]
    y_tr  = y_dir[:train_end]
    y_val = y_dir[test_start:test_end]

    scaler       = StandardScaler()
    X_tr_scaled  = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    log_reg = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    log_reg.fit(X_tr_scaled, y_tr)
    preds = log_reg.predict(X_val_scaled)

    log_preds.extend(preds)
    log_true.extend(y_val)
    acc = accuracy_score(y_val, preds)
    print(f"  Fold {fold_idx+1}: {acc*100:.2f}%")

log_preds = np.array(log_preds)
log_true  = np.array(log_true)

saved_preds["logistic"] = log_preds
saved_true["logistic"]  = log_true

log_acc = accuracy_score(log_true, log_preds)
print(f"Logistic Accuracy: {log_acc*100:.2f}%")

# =============================================================================
# MODEL 4: LSTM — save predictions
# =============================================================================
print("\n--- Re-running: LSTM ---")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

LOOKBACK = 20

def create_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

scaler_lstm  = StandardScaler()
X_scaled     = scaler_lstm.fit_transform(X_raw)
X_seq, y_seq = create_sequences(X_scaled, y_dir, LOOKBACK)
seq_folds    = get_folds(len(X_seq), n_splits=3)

lstm_preds = []
lstm_true  = []

for fold_idx, (_, train_end, test_start, test_end) in enumerate(seq_folds):
    X_tr  = X_seq[:train_end]
    X_val = X_seq[test_start:test_end]
    y_tr  = y_seq[:train_end]
    y_val = y_seq[test_start:test_end]

    model_lstm = build_lstm((LOOKBACK, len(feature_cols)))
    early_stop = EarlyStopping(monitor="val_loss", patience=15,
                               restore_best_weights=True, verbose=0)
    model_lstm.fit(X_tr, y_tr, epochs=100, batch_size=32,
                   validation_data=(X_val, y_val),
                   callbacks=[early_stop], verbose=0)

    preds_prob = model_lstm.predict(X_val, verbose=0).flatten()
    preds      = (preds_prob > 0.5).astype(int)

    lstm_preds.extend(preds)
    lstm_true.extend(y_val)
    acc = accuracy_score(y_val, preds)
    print(f"  Fold {fold_idx+1}: {acc*100:.2f}%")

lstm_preds = np.array(lstm_preds)
lstm_true  = np.array(lstm_true)

saved_preds["lstm"] = lstm_preds
saved_true["lstm"]  = lstm_true

lstm_acc = accuracy_score(lstm_true, lstm_preds)
print(f"LSTM Accuracy: {lstm_acc*100:.2f}%")

# --- Save all prediction arrays ---
preds_dir = os.path.join(script_dir, "..", "results", "predictions")
os.makedirs(preds_dir, exist_ok=True)

for model_name, preds in saved_preds.items():
    pd.DataFrame({
        "predictions": preds,
        "true":        saved_true[model_name]
    }).to_csv(os.path.join(preds_dir, f"sp500_{model_name}_preds.csv"),
              index=False)

print(f"\nSaved prediction arrays to results/predictions/")

# =============================================================================
# SECTION 2: DIEBOLD-MARIANO TESTS
# =============================================================================
# Tests whether accuracy differences between model pairs are significant.
# Uses squared error loss: loss(t) = (pred(t) - true(t))^2
# For binary predictions: error = 1 if wrong, 0 if correct.
# Loss differential d(t) = loss_model1(t) - loss_model2(t)
# DM statistic: mean(d) / sqrt(variance(d) / n)
# p < 0.05: one model is significantly better.
# p > 0.05: difference is not statistically distinguishable from noise.
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: DIEBOLD-MARIANO TESTS")
print("=" * 60)

def diebold_mariano(preds1, preds2, true_labels):
    """
    Diebold-Mariano test for binary direction predictions.

    Parameters:
        preds1, preds2 : prediction arrays from two models
        true_labels    : actual direction labels (0 or 1)

    Returns:
        dm_stat : DM test statistic
        p_value : two-sided p-value
        winner  : which model has lower loss (better accuracy)

    Loss function: 0/1 loss (1 if wrong, 0 if correct).
    Loss differential d(t) = loss1(t) - loss2(t).
    Positive mean_d: model 1 has higher loss → model 2 is better.
    Negative mean_d: model 2 has higher loss → model 1 is better.
    """
    loss1 = (preds1 != true_labels).astype(float)  # 1 if wrong
    loss2 = (preds2 != true_labels).astype(float)  # 1 if wrong
    d     = loss1 - loss2
    n     = len(d)

    mean_d  = np.mean(d)
    # Newey-West variance with lag=1 for one-step-ahead forecasts
    gamma_0 = np.var(d, ddof=1)
    if n > 1:
        gamma_1 = np.cov(d[1:], d[:-1], ddof=1)[0, 1]
    else:
        gamma_1 = 0
    nw_var  = gamma_0 + 2 * gamma_1

    if nw_var <= 0:
        return np.nan, np.nan, "inconclusive"

    dm_stat = mean_d / np.sqrt(nw_var / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    if mean_d > 0:
        winner = "Model 2 (lower loss)"
    elif mean_d < 0:
        winner = "Model 1 (lower loss)"
    else:
        winner = "Tie"

    return dm_stat, p_value, winner

# Run DM tests on key pairs
dm_comparisons = [
    ("Logistic vs Naive",  "logistic", "naive"),
    ("LSTM vs Naive",      "lstm",     "naive"),
    ("ARIMA vs Naive",     "arima",    "naive"),
    ("LSTM vs Logistic",   "lstm",     "logistic"),
    ("LSTM vs ARIMA",      "lstm",     "arima"),
    ("Logistic vs ARIMA",  "logistic", "arima"),
]

dm_results = []

print(f"\n{'Comparison':<25} {'DM Stat':>10} {'p-value':>10} "
      f"{'Significant':>12} {'Better Model':>20}")
print("-" * 80)

for label, m1, m2 in dm_comparisons:
    # Align array lengths — LSTM has fewer predictions due to lookback window
    preds1 = saved_preds[m1]
    preds2 = saved_preds[m2]
    true1  = saved_true[m1]

    min_len = min(len(preds1), len(preds2))
    preds1  = preds1[-min_len:]
    preds2  = preds2[-min_len:]
    true1   = true1[-min_len:]

    dm_stat, p_val, winner = diebold_mariano(preds1, preds2, true1)
    

    significant = "YES ✓" if p_val < 0.05 else "NO"

    # Determine which model name is better for display
    if winner == "Model 1 (lower loss)":
        better = m1.capitalize()
    elif winner == "Model 2 (lower loss)":
        better = m2.capitalize()
    else:
        better = "Tie"

    print(f"{label:<25} {dm_stat:>10.4f} {p_val:>10.4f} "
          f"{significant:>12} {better:>20}")

    dm_results.append({
        "comparison":   label,
        "model_1":      m1,
        "model_2":      m2,
        "dm_statistic": round(dm_stat, 4) if not np.isnan(dm_stat) else None,
        "p_value":      round(p_val, 4)   if not np.isnan(p_val)   else None,
        "significant":  p_val < 0.05      if not np.isnan(p_val)   else None,
        "better_model": better,
    })

# =============================================================================
# SECTION 3: BINOMIAL SIGNIFICANCE TESTS
# =============================================================================
# Tests whether each model's accuracy is significantly above the naive
# baseline. Uses one-sided binomial test.
# H0: model accuracy <= naive baseline accuracy
# H1: model accuracy > naive baseline accuracy
# p < 0.05: model significantly beats the baseline.
# p > 0.05: cannot conclude model beats the baseline.
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: BINOMIAL SIGNIFICANCE TESTS")
print("=" * 60)

# All results across all assets
# Format: (asset, model, accuracy, n_total, baseline_accuracy)
all_results = [
    # S&P 500
    ("S&P 500", "ARIMA",    0.5073, len(arima_true),  0.5393),
    ("S&P 500", "Logistic", 0.5440, len(log_true),    0.5393),
    ("S&P 500", "LSTM",     0.5440, len(lstm_true),   0.5393),
    ("S&P 500", "GRU",      0.5466, len(lstm_true),   0.5393),
    # NVDA
    ("NVDA", "GBM+MC",   0.5084, 791*3, 0.5424),
    ("NVDA", "Logistic", 0.5348, 791*3, 0.5424),
    ("NVDA", "RF",       0.5361, 791*3, 0.5424),
    ("NVDA", "LSTM",     0.5314, 791*3, 0.5424),
    ("NVDA", "GRU",      0.5280, 791*3, 0.5424),
    # GLD
    ("GLD", "GBM+MC",   0.5245, 791*3, 0.5435),
    ("GLD", "Logistic", 0.5404, 791*3, 0.5435),
    ("GLD", "RF",       0.5214, 791*3, 0.5435),
    ("GLD", "LSTM",     0.5414, 791*3, 0.5435),
    ("GLD", "GRU",      0.5419, 791*3, 0.5435),
    # EWJ
    ("EWJ", "GBM+MC",   0.4803, 791*3, 0.5273),
    ("EWJ", "Logistic", 0.5059, 791*3, 0.5273),
    ("EWJ", "RF",       0.4958, 791*3, 0.5273),
    ("EWJ", "LSTM",     0.5327, 791*3, 0.5273),
    ("EWJ", "GRU",      0.5247, 791*3, 0.5273),
]

binomial_results = []

print(f"\n{'Asset':<10} {'Model':<12} {'Accuracy':>10} {'Baseline':>10} "
      f"{'p-value':>10} {'Significant':>12}")
print("-" * 70)

for asset, model, acc, n_total, baseline in all_results:
    n_correct = int(acc * n_total)

    # One-sided binomial test: is accuracy > baseline?
    result  = stats.binomtest(n_correct, n_total, p=baseline,
                               alternative="greater")
    p_val   = result.pvalue
    sig     = "YES ✓" if p_val < 0.05 else "NO"

    print(f"{asset:<10} {model:<12} {acc*100:>9.2f}% {baseline*100:>9.2f}% "
          f"{p_val:>10.4f} {sig:>12}")

    binomial_results.append({
        "asset":       asset,
        "model":       model,
        "accuracy":    acc,
        "baseline":    baseline,
        "n_total":     n_total,
        "n_correct":   n_correct,
        "p_value":     round(p_val, 4),
        "significant": p_val < 0.05,
    })

# =============================================================================
# SECTION 4: SAVE ALL RESULTS
# =============================================================================

results_dir = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)

dm_df = pd.DataFrame(dm_results)
dm_df.to_csv(os.path.join(results_dir, "dm_test_results.csv"), index=False)
print(f"\nSaved DM results to results/dm_test_results.csv")

binomial_df = pd.DataFrame(binomial_results)
binomial_df.to_csv(os.path.join(results_dir, "binomial_test_results.csv"),
                   index=False)
print(f"Saved binomial results to results/binomial_test_results.csv")

# =============================================================================
# SECTION 5: VISUALIZATIONS
# =============================================================================

figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Part 15: Significance Testing Results",
             fontsize=14, fontweight="bold", color="white")

# --- Plot 1: DM Test p-values ---
ax1 = axes[0]
dm_labels  = [r["comparison"] for r in dm_results]
dm_pvalues = [r["p_value"] if r["p_value"] is not None else 1.0
              for r in dm_results]
dm_colors  = ["#5ce0a0" if p < 0.05 else "#e05c5c" for p in dm_pvalues]

bars = ax1.barh(dm_labels, dm_pvalues, color=dm_colors, alpha=0.85,
                edgecolor="white")
ax1.axvline(x=0.05, color="white", linestyle="--", linewidth=1.5,
            label="p=0.05 threshold")
ax1.set_xlabel("p-value", color="white")
ax1.set_title("Diebold-Mariano Test p-values\n(green = significant)",
              color="white")
ax1.legend(fontsize=9)
ax1.tick_params(colors="white")
ax1.set_xlim(0, 1)

for bar, p in zip(bars, dm_pvalues):
    ax1.text(min(p + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
             f"{p:.3f}", va="center", fontsize=8, color="white")

# --- Plot 2: Binomial test p-values by asset ---
ax2 = axes[1]

sp500_results = [r for r in binomial_results if r["asset"] == "S&P 500"]
labels  = [r["model"] for r in sp500_results]
pvalues = [r["p_value"] for r in sp500_results]
colors  = ["#5ce0a0" if p < 0.05 else "#e05c5c" for p in pvalues]

bars2 = ax2.bar(labels, pvalues, color=colors, alpha=0.85, edgecolor="white")
ax2.axhline(y=0.05, color="white", linestyle="--", linewidth=1.5,
            label="p=0.05 threshold")
ax2.set_ylabel("p-value", color="white")
ax2.set_title("Binomial Test p-values\nS&P 500 Models vs Naive Baseline",
              color="white")
ax2.legend(fontsize=9)
ax2.tick_params(colors="white")
ax2.set_ylim(0, 1)

for bar, p in zip(bars2, pvalues):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02,
             f"{p:.3f}", ha="center", fontsize=8, color="white")

plt.tight_layout()
fig_path = os.path.join(figures_dir, "part15_significance_tests.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to: {fig_path}")

# =============================================================================
# SECTION 6: SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SIGNIFICANCE TESTING SUMMARY")
print("=" * 60)

sig_dm  = sum(1 for r in dm_results
              if r["p_value"] is not None and r["p_value"] < 0.05)
sig_bin = sum(1 for r in binomial_results if r["significant"])

print(f"Diebold-Mariano tests run:      {len(dm_results)}")
print(f"DM tests significant (p<0.05):  {sig_dm}")
print(f"Binomial tests run:             {len(binomial_results)}")
print(f"Binomial tests significant:     {sig_bin}")
print()
print("Key finding:")
if sig_dm == 0 and sig_bin == 0:
    print("NO model difference is statistically significant.")
    print("All accuracy differences are consistent with random noise.")
    print("This strongly supports the EMH and confirms the null result.")
elif sig_bin > 0 and sig_dm == 0:
    print("Some binomial tests significant but no DM tests significant.")
    print("Models may beat their own baseline marginally but differences")
    print("between models are indistinguishable from noise.")
else:
    print(f"{sig_dm} DM comparisons and {sig_bin} binomial tests significant.")
    print("Check dm_test_results.csv and binomial_test_results.csv for details.")