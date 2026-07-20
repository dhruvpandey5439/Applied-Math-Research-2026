
# PART 16: REGIME SPLIT ANALYSIS

# What this part does:
# Splits the S&P 500 data into two market regimes and re-runs key models
# on each regime separately:
#   Pre-2020:  2000-01-01 to 2019-12-31 — stable bull market, low volatility
#   Post-2020: 2020-01-01 to 2026-05-28 — COVID crash, high volatility,
#              AI boom, rate hikes, geopolitical uncertainty
#
# Why this matters:
# A model that only works in one market regime is not generalizable.
# If findings are consistent across both regimes, they are more reliable.
# If findings differ significantly between regimes, that is itself
# a finding worth discussing in the paper — regime-dependence is real
# and important to report honestly.
#
# Models tested in each regime:
#   1. Naive Baseline
#   2. Logistic Regression
#   3. Random Forest
#   4. LSTM


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

script_dir = os.path.dirname(os.path.abspath(__file__))


# STEP 1: LOAD DATA


features_path = os.path.join(
    script_dir, "..", "data processed", "sp500_features.csv"
)
data = pd.read_csv(features_path, index_col=0, parse_dates=True)
data = data.dropna()

print(f"Full data: {data.index[0].date()} to {data.index[-1].date()} "
      f"({len(data)} rows)")

feature_cols = [
    "return_lag_1", "return_lag_2", "return_lag_3",
    "return_lag_5", "return_ma_5",  "return_ma_10",
    "volatility_5", "volatility_10", "volatility_20"
]


# STEP 2: SPLIT INTO REGIMES

# Pre-2020: everything before COVID crash
# Post-2020: COVID crash onwards — fundamentally different market
# The split date of 2020-01-01 is standard in financial research
# as a structural break point due to COVID-19.


pre_2020  = data[data.index < "2020-01-01"].copy()
post_2020 = data[data.index >= "2020-01-01"].copy()

print(f"Pre-2020:  {pre_2020.index[0].date()} to "
      f"{pre_2020.index[-1].date()} ({len(pre_2020)} rows)")
print(f"Post-2020: {post_2020.index[0].date()} to "
      f"{post_2020.index[-1].date()} ({len(post_2020)} rows)\n")


# HELPER FUNCTIONS


LOOKBACK = 20

def get_folds(n, n_splits=3, test_frac=0.20, min_train_frac=0.40):
    """
    Walk-forward folds. Using larger test_frac=0.20 for regime subsets
    since they have fewer rows than the full dataset.
    """
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


def run_regime(regime_data, regime_name):
    """
    Runs all 4 models on a given regime subset.
    Returns a dictionary of results.
    """
    print(f"\n{'='*60}")
    print(f"REGIME: {regime_name}")
    print(f"{'='*60}")

    X_raw = regime_data[feature_cols].values
    y_dir = regime_data["target_direction"].values
    n     = len(regime_data)
    folds = get_folds(n, n_splits=3)

    if len(folds) == 0:
        print(f"Not enough data for walk-forward validation in {regime_name}")
        return None

    results = {"regime": regime_name}

    # --- Naive Baseline ---
    print(f"\n--- {regime_name}: Naive Baseline ---")
    naive_preds = []
    naive_true  = []
    for (_, train_end, test_start, test_end) in folds:
        y_val = y_dir[test_start:test_end]
        preds = np.ones(len(y_val), dtype=int)
        naive_preds.extend(preds)
        naive_true.extend(y_val)

    naive_acc = accuracy_score(naive_true, naive_preds)
    naive_f1  = f1_score(naive_true, naive_preds, zero_division=0)
    print(f"Naive Accuracy: {naive_acc*100:.2f}% | F1: {naive_f1:.4f}")
    results["naive_accuracy"] = naive_acc
    results["naive_f1"]       = naive_f1

    # --- Logistic Regression ---
    print(f"\n--- {regime_name}: Logistic Regression ---")
    log_fold_accs = []
    all_log_preds = []
    all_log_true  = []

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

        acc = accuracy_score(y_val, preds)
        log_fold_accs.append(acc)
        all_log_preds.extend(preds)
        all_log_true.extend(y_val)
        print(f"  Fold {fold_idx+1}: {acc*100:.2f}%")

    log_acc = np.mean(log_fold_accs)
    log_f1  = f1_score(all_log_true, all_log_preds, zero_division=0)
    print(f"Logistic Mean: {log_acc*100:.2f}% | F1: {log_f1:.4f}")
    results["logistic_accuracy"] = log_acc
    results["logistic_f1"]       = log_f1

    # --- Random Forest ---
    print(f"\n--- {regime_name}: Random Forest ---")
    rf_fold_accs = []
    all_rf_preds = []
    all_rf_true  = []

    for fold_idx, (_, train_end, test_start, test_end) in enumerate(folds):
        X_tr  = X_raw[:train_end]
        X_val = X_raw[test_start:test_end]
        y_tr  = y_dir[:train_end]
        y_val = y_dir[test_start:test_end]

        scaler       = StandardScaler()
        X_tr_scaled  = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_tr_scaled, y_tr)
        preds = rf.predict(X_val_scaled)

        acc = accuracy_score(y_val, preds)
        rf_fold_accs.append(acc)
        all_rf_preds.extend(preds)
        all_rf_true.extend(y_val)
        print(f"  Fold {fold_idx+1}: {acc*100:.2f}%")

    rf_acc = np.mean(rf_fold_accs)
    rf_f1  = f1_score(all_rf_true, all_rf_preds, zero_division=0)
    print(f"RF Mean: {rf_acc*100:.2f}% | F1: {rf_f1:.4f}")
    results["rf_accuracy"] = rf_acc
    results["rf_f1"]       = rf_f1

    # --- LSTM ---
    print(f"\n--- {regime_name}: LSTM ---")

    scaler_lstm  = StandardScaler()
    X_scaled     = scaler_lstm.fit_transform(X_raw)
    X_seq, y_seq = create_sequences(X_scaled, y_dir, LOOKBACK)
    seq_folds    = get_folds(len(X_seq), n_splits=3)

    lstm_fold_accs = []
    all_lstm_preds = []
    all_lstm_true  = []

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

        acc = accuracy_score(y_val, preds)
        lstm_fold_accs.append(acc)
        all_lstm_preds.extend(preds)
        all_lstm_true.extend(y_val)
        print(f"  Fold {fold_idx+1}: {acc*100:.2f}%")

    lstm_acc = np.mean(lstm_fold_accs)
    lstm_f1  = f1_score(all_lstm_true, all_lstm_preds, zero_division=0)
    print(f"LSTM Mean: {lstm_acc*100:.2f}% | F1: {lstm_f1:.4f}")
    results["lstm_accuracy"] = lstm_acc
    results["lstm_f1"]       = lstm_f1

    # Summary
    print(f"\n{regime_name} SUMMARY:")
    print(f"  Naive:    {naive_acc*100:.2f}%")
    print(f"  Logistic: {log_acc*100:.2f}% "
          f"({'BEAT' if log_acc > naive_acc else 'BELOW'} floor by "
          f"{abs(log_acc-naive_acc)*100:.2f}pp)")
    print(f"  RF:       {rf_acc*100:.2f}% "
          f"({'BEAT' if rf_acc > naive_acc else 'BELOW'} floor by "
          f"{abs(rf_acc-naive_acc)*100:.2f}pp)")
    print(f"  LSTM:     {lstm_acc*100:.2f}% "
          f"({'BEAT' if lstm_acc > naive_acc else 'BELOW'} floor by "
          f"{abs(lstm_acc-naive_acc)*100:.2f}pp)")

    return results



# STEP 3: RUN BOTH REGIMES


pre_results  = run_regime(pre_2020,  "Pre-2020  (2000-2019)")
post_results = run_regime(post_2020, "Post-2020 (2020-2026)")


# STEP 4: SAVE RESULTS


results_dir = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)

regime_df = pd.DataFrame([pre_results, post_results])
regime_df.to_csv(os.path.join(results_dir, "regime_analysis.csv"), index=False)
print(f"\nSaved regime results to results/regime_analysis.csv")


# STEP 5: CROSS-REGIME COMPARISON FIGURE


figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Part 16: Regime Split Analysis — Pre vs Post 2020",
             fontsize=13, fontweight="bold", color="white")

model_keys   = ["naive", "logistic", "rf", "lstm"]
model_labels = ["Naive", "Logistic", "RF", "LSTM"]
bar_colors   = ["#888888", "#5ce0a0", "#5cb8e0", "#a05ce0"]

for ax, results, title in [
    (axes[0], pre_results,  "Pre-2020 (2000-2019)"),
    (axes[1], post_results, "Post-2020 (2020-2026)")
]:
    if results is None:
        continue

    accs  = [results[f"{k}_accuracy"] * 100 for k in model_keys]
    floor = results["naive_accuracy"] * 100

    bars = ax.bar(model_labels, accs, color=bar_colors,
                  alpha=0.85, edgecolor="white")
    ax.axhline(floor, color="white", linestyle="--",
               linewidth=1.5, label=f"Naive floor: {floor:.1f}%")
    ax.axhline(50, color="gray", linestyle=":", linewidth=1,
               label="50% random chance")
    ax.set_title(title, color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("Direction Accuracy (%)", color="white")
    ax.set_ylim(40, 70)
    ax.legend(fontsize=9)
    ax.tick_params(colors="white")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom",
                fontsize=9, color="white")

plt.tight_layout()
fig_path = os.path.join(figures_dir, "part16_regime_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to: {fig_path}")


# STEP 6: FINAL SUMMARY


print("\n" + "=" * 60)
print("REGIME ANALYSIS SUMMARY")
print("=" * 60)

if pre_results and post_results:
    print(f"\n{'Model':<15} {'Pre-2020':>12} {'Post-2020':>12} {'Difference':>12}")
    print("-" * 55)
    for key, label in [("naive",    "Naive"),
                        ("logistic", "Logistic"),
                        ("rf",       "RF"),
                        ("lstm",     "LSTM")]:
        pre_acc  = pre_results[f"{key}_accuracy"]  * 100
        post_acc = post_results[f"{key}_accuracy"] * 100
        diff     = post_acc - pre_acc
        print(f"{label:<15} {pre_acc:>11.2f}% {post_acc:>11.2f}% "
              f"{diff:>+11.2f}pp")

    print("\nKey question: do findings hold across both regimes?")
    pre_log_beats  = pre_results["logistic_accuracy"]  > pre_results["naive_accuracy"]
    post_log_beats = post_results["logistic_accuracy"] > post_results["naive_accuracy"]
    pre_lstm_beats  = pre_results["lstm_accuracy"]  > pre_results["naive_accuracy"]
    post_lstm_beats = post_results["lstm_accuracy"] > post_results["naive_accuracy"]

    print(f"Logistic beats naive — Pre-2020:  {'YES' if pre_log_beats  else 'NO'}")
    print(f"Logistic beats naive — Post-2020: {'YES' if post_log_beats else 'NO'}")
    print(f"LSTM beats naive     — Pre-2020:  {'YES' if pre_lstm_beats  else 'NO'}")
    print(f"LSTM beats naive     — Post-2020: {'YES' if post_lstm_beats else 'NO'}")