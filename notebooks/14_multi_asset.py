# =============================================================================
# PART 14: MULTI-ASSET EXPANSION
# =============================================================================
# Research Question:
# To what extent do increasingly complex machine learning models improve
# predictive performance compared to classical statistical and mathematical
# methods when forecasting financial time-series data?
#
# What this part does:
# Runs the full pipeline on 3 additional assets beyond S&P 500:
#   NVDA  — individual stock (NVIDIA Corporation)
#   GLD   — commodity (Gold ETF)
#   EWJ   — international index (Nikkei 225 ETF)
#
# Models run on each asset:
#   1. Naive Baseline (Always Up)  — the floor
#   2. GBM + Monte Carlo           — mathematical tradition
#   3. Logistic Regression         — simple ML
#   4. Random Forest               — mid-complexity ML
#   5. LSTM                        — deep learning
#   6. GRU                         — deep learning comparison
#
# Why no ARIMA:
# ARIMA underperformed on S&P 500 (-3.20pp vs naive baseline) and
# requires statsmodels which has a version conflict in tf_env.
# Its classical statistical limitation is already established on
# the primary asset. The paper will state this explicitly.
#
# Why GBM + Monte Carlo is included:
# GBM has no external dependencies beyond numpy — no version conflicts.
# It represents the mathematical tradition needed for the three-way
# comparison across all assets.
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"]  = "#0e0e0e"
plt.rcParams["axes.facecolor"]    = "#1a1a1a"
plt.rcParams["savefig.facecolor"] = "#0e0e0e"

# =============================================================================
# CONFIGURATION
# =============================================================================

TICKERS        = ["NVDA", "GLD", "EWJ"]
START_DATE     = "2000-01-01"
END_DATE       = "2026-06-01"
LOOKBACK       = 20
N_SPLITS       = 3
N_SIMULATIONS  = 1000   # GBM Monte Carlo paths per forecast
SP500_BASELINE = 0.5393

script_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def download_and_prepare(ticker, start, end):
    """
    Downloads price data and engineers features + targets.
    Identical pipeline to Parts 1-4 applied to any asset.
    """
    print(f"\nDownloading {ticker}...")
    raw = yf.download(ticker, start=start, end=end, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if len(raw) < 500:
        print(f"WARNING: {ticker} has fewer than 500 rows. Skipping.")
        return None

    raw["Return"] = np.log(raw["Close"] / raw["Close"].shift(1))
    raw = raw.dropna()

    raw["return_lag_1"]  = raw["Return"].shift(1)
    raw["return_lag_2"]  = raw["Return"].shift(2)
    raw["return_lag_3"]  = raw["Return"].shift(3)
    raw["return_lag_5"]  = raw["Return"].shift(5)
    raw["return_ma_5"]   = raw["Return"].rolling(5,  min_periods=5).mean().shift(1)
    raw["return_ma_10"]  = raw["Return"].rolling(10, min_periods=10).mean().shift(1)
    raw["volatility_5"]  = raw["Return"].rolling(5,  min_periods=5).std().shift(1)
    raw["volatility_10"] = raw["Return"].rolling(10, min_periods=10).std().shift(1)
    raw["volatility_20"] = raw["Return"].rolling(20, min_periods=20).std().shift(1)

    raw["target_direction"] = (raw["Return"].shift(-1) > 0).astype(int)
    raw["target_return"]    = raw["Return"].shift(-1)

    raw = raw.dropna()

    print(f"{ticker}: {raw.index[0].date()} to {raw.index[-1].date()} "
          f"({len(raw)} rows)")

    return raw


def get_folds(n, n_splits, test_frac=0.12, min_train_frac=0.50):
    """
    Creates walk-forward fold indices.
    Same logic as Parts 12/13.
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
    """
    Creates overlapping sequences for LSTM/GRU.
    Identical to Parts 12/13.
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def build_lstm(input_shape):
    """
    Builds LSTM model — identical architecture to Part 12.
    """
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


def build_gru(input_shape):
    """
    Builds GRU model — identical architecture to Part 13.
    """
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def simulate_gbm_direction(mu, sigma, S0, n_simulations=1000, dt=1):
    """
    Simulates N price paths one step forward using GBM.
    Identical to Part 9.
    Returns direction (1=up, 0=down) based on median simulated price.
    """
    Z                = np.random.standard_normal(n_simulations)
    simulated_prices = S0 * np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    )
    median_price = np.median(simulated_prices)
    direction    = 1 if median_price > S0 else 0
    return direction


# =============================================================================
# MAIN PIPELINE
# =============================================================================

feature_cols = [
    "return_lag_1", "return_lag_2", "return_lag_3",
    "return_lag_5", "return_ma_5",  "return_ma_10",
    "volatility_5", "volatility_10", "volatility_20"
]

all_asset_results = []

for ticker in TICKERS:

    print(f"\n{'='*60}")
    print(f"RUNNING PIPELINE: {ticker}")
    print(f"{'='*60}")

    data = download_and_prepare(ticker, START_DATE, END_DATE)
    if data is None:
        continue

    asset_data_dir = os.path.join(script_dir, "..", "data processed", ticker)
    os.makedirs(asset_data_dir, exist_ok=True)
    data.to_csv(os.path.join(asset_data_dir, f"{ticker}_features.csv"))

    X_raw          = data[feature_cols].values
    y_dir          = data["target_direction"].values
    returns        = data["Return"].values
    prices         = data["Close"].values
    n              = len(data)
    folds          = get_folds(n, N_SPLITS)
    ticker_results = {"ticker": ticker}

    # -------------------------------------------------------------------------
    # MODEL 1: NAIVE BASELINE
    # -------------------------------------------------------------------------
    print(f"\n--- {ticker}: Naive Baseline ---")

    all_naive_preds = []
    all_naive_true  = []

    for (_, train_end, test_start, test_end) in folds:
        y_val = y_dir[test_start:test_end]
        preds = np.ones(len(y_val), dtype=int)
        all_naive_preds.extend(preds)
        all_naive_true.extend(y_val)

    naive_acc = accuracy_score(all_naive_true, all_naive_preds)
    naive_f1  = f1_score(all_naive_true, all_naive_preds, zero_division=0)

    print(f"Always Up Accuracy: {naive_acc*100:.2f}% | F1: {naive_f1:.4f}")
    ticker_results["naive_accuracy"] = naive_acc
    ticker_results["naive_f1"]       = naive_f1

    # -------------------------------------------------------------------------
    # MODEL 2: GBM + MONTE CARLO
    # -------------------------------------------------------------------------
    print(f"\n--- {ticker}: GBM + Monte Carlo ---")

    gbm_fold_accs = []
    all_gbm_preds = []
    all_gbm_true  = []

    for fold_idx, (_, train_end, test_start, test_end) in enumerate(folds):
        fold_train_ret = returns[:train_end]
        fold_val_ret   = returns[test_start:test_end]
        fold_true      = y_dir[test_start:test_end]
        fold_prices    = prices[test_start:test_end]

        fold_preds = []

        for i in range(len(fold_val_ret)):
            history = np.concatenate([fold_train_ret, fold_val_ret[:i]])
            mu      = np.mean(history)
            sigma   = np.std(history)
            S0      = fold_prices[i]

            direction = simulate_gbm_direction(
                mu=mu, sigma=sigma, S0=S0,
                n_simulations=N_SIMULATIONS
            )
            fold_preds.append(direction)

        acc = accuracy_score(fold_true, fold_preds)
        gbm_fold_accs.append(acc)
        all_gbm_preds.extend(fold_preds)
        all_gbm_true.extend(fold_true)
        print(f"  Fold {fold_idx+1}: {acc*100:.2f}%")

    gbm_acc = np.mean(gbm_fold_accs)
    gbm_f1  = f1_score(all_gbm_true, all_gbm_preds, zero_division=0)
    print(f"GBM Mean Accuracy: {gbm_acc*100:.2f}% | F1: {gbm_f1:.4f}")
    ticker_results["gbm_accuracy"] = gbm_acc
    ticker_results["gbm_f1"]       = gbm_f1

    # -------------------------------------------------------------------------
    # MODEL 3: LOGISTIC REGRESSION
    # -------------------------------------------------------------------------
    print(f"\n--- {ticker}: Logistic Regression ---")

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
    print(f"Logistic Mean Accuracy: {log_acc*100:.2f}% | F1: {log_f1:.4f}")
    ticker_results["logistic_accuracy"] = log_acc
    ticker_results["logistic_f1"]       = log_f1

    # -------------------------------------------------------------------------
    # MODEL 4: RANDOM FOREST
    # -------------------------------------------------------------------------
    print(f"\n--- {ticker}: Random Forest ---")

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
    print(f"Random Forest Mean Accuracy: {rf_acc*100:.2f}% | F1: {rf_f1:.4f}")
    ticker_results["rf_accuracy"] = rf_acc
    ticker_results["rf_f1"]       = rf_f1

    # -------------------------------------------------------------------------
    # MODEL 5: LSTM
    # -------------------------------------------------------------------------
    print(f"\n--- {ticker}: LSTM ---")

    scaler_seq   = StandardScaler()
    X_scaled     = scaler_seq.fit_transform(X_raw)
    X_seq, y_seq = create_sequences(X_scaled, y_dir, LOOKBACK)
    seq_folds    = get_folds(len(X_seq), N_SPLITS)

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
    print(f"LSTM Mean Accuracy: {lstm_acc*100:.2f}% | F1: {lstm_f1:.4f}")
    ticker_results["lstm_accuracy"] = lstm_acc
    ticker_results["lstm_f1"]       = lstm_f1

    # -------------------------------------------------------------------------
    # MODEL 6: GRU
    # -------------------------------------------------------------------------
    print(f"\n--- {ticker}: GRU ---")

    gru_fold_accs = []
    all_gru_preds = []
    all_gru_true  = []

    for fold_idx, (_, train_end, test_start, test_end) in enumerate(seq_folds):
        X_tr  = X_seq[:train_end]
        X_val = X_seq[test_start:test_end]
        y_tr  = y_seq[:train_end]
        y_val = y_seq[test_start:test_end]

        model_gru  = build_gru((LOOKBACK, len(feature_cols)))
        early_stop = EarlyStopping(monitor="val_loss", patience=15,
                                   restore_best_weights=True, verbose=0)
        model_gru.fit(X_tr, y_tr, epochs=100, batch_size=32,
                      validation_data=(X_val, y_val),
                      callbacks=[early_stop], verbose=0)

        preds_prob = model_gru.predict(X_val, verbose=0).flatten()
        preds      = (preds_prob > 0.5).astype(int)

        acc = accuracy_score(y_val, preds)
        gru_fold_accs.append(acc)
        all_gru_preds.extend(preds)
        all_gru_true.extend(y_val)
        print(f"  Fold {fold_idx+1}: {acc*100:.2f}%")

    gru_acc = np.mean(gru_fold_accs)
    gru_f1  = f1_score(all_gru_true, all_gru_preds, zero_division=0)
    print(f"GRU Mean Accuracy: {gru_acc*100:.2f}% | F1: {gru_f1:.4f}")
    ticker_results["gru_accuracy"] = gru_acc
    ticker_results["gru_f1"]       = gru_f1

    all_asset_results.append(ticker_results)

    # -------------------------------------------------------------------------
    # SAVE ASSET RESULTS
    # -------------------------------------------------------------------------
    asset_results_dir = os.path.join(script_dir, "..", "results", ticker)
    os.makedirs(asset_results_dir, exist_ok=True)

    asset_df = pd.DataFrame([{
        "model":             m,
        "accuracy":          ticker_results[f"{k}_accuracy"],
        "f1_score":          ticker_results[f"{k}_f1"],
        "vs_asset_floor":    ticker_results[f"{k}_accuracy"] - ticker_results["naive_accuracy"],
        "vs_sp500_baseline": ticker_results[f"{k}_accuracy"] - SP500_BASELINE,
    } for m, k in [
        ("Naive Baseline (Always Up)", "naive"),
        ("GBM + Monte Carlo",          "gbm"),
        ("Logistic Regression",        "logistic"),
        ("Random Forest",              "rf"),
        ("LSTM",                       "lstm"),
        ("GRU",                        "gru"),
    ]])

    asset_df.to_csv(os.path.join(asset_results_dir,
                                  f"{ticker}_model_comparison.csv"), index=False)
    print(f"\nSaved {ticker} results.")

    # -------------------------------------------------------------------------
    # ASSET FIGURE
    # -------------------------------------------------------------------------
    asset_figures_dir = os.path.join(script_dir, "..", "figures", ticker)
    os.makedirs(asset_figures_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(f"Part 14: {ticker} — Model Comparison",
                 fontsize=13, fontweight="bold", color="white")

    model_names = ["Naive\nBaseline", "GBM+\nMC",
                   "Logistic\nRegression", "Random\nForest",
                   "LSTM", "GRU"]
    model_accs  = [naive_acc, gbm_acc, log_acc, rf_acc, lstm_acc, gru_acc]
    bar_colors  = ["#888888", "#e0c45c", "#5ce0a0",
                   "#5cb8e0", "#a05ce0", "#e05c8a"]

    bars = ax.bar(model_names, [a*100 for a in model_accs],
                  color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(naive_acc * 100, color="white", linestyle="--",
               linewidth=1.5, label=f"{ticker} floor ({naive_acc*100:.1f}%)")
    ax.axhline(SP500_BASELINE * 100, color="yellow", linestyle=":",
               linewidth=1.2, label="S&P 500 floor (53.9%)")
    ax.axhline(50, color="gray", linestyle=":", linewidth=1,
               label="50% random chance")
    ax.set_ylabel("Direction Accuracy (%)", color="white")
    ax.set_title(f"{ticker}: All 6 Models", color="white")
    ax.set_ylim(40, 75)
    ax.legend(fontsize=9)
    ax.tick_params(colors="white")

    for bar, acc in zip(bars, model_accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{acc*100:.1f}%", ha="center", va="bottom",
                fontsize=9, color="white")

    plt.tight_layout()
    fig_path = os.path.join(asset_figures_dir,
                             f"{ticker}_model_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {fig_path}")

# =============================================================================
# CROSS-ASSET SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print("CROSS-ASSET SUMMARY")
print(f"{'='*60}")

summary_df = pd.DataFrame(all_asset_results).set_index("ticker")
print(summary_df[[
    "naive_accuracy", "gbm_accuracy", "logistic_accuracy",
    "rf_accuracy", "lstm_accuracy", "gru_accuracy"
]].round(4).to_string())

summary_path = os.path.join(script_dir, "..", "results",
                             "cross_asset_summary.csv")
summary_df.to_csv(summary_path)
print(f"\nSaved cross-asset summary to: {summary_path}")

# =============================================================================
# CROSS-ASSET FIGURE
# =============================================================================

figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

fig, axes = plt.subplots(1, len(all_asset_results),
                          figsize=(8 * len(all_asset_results), 6))

if len(all_asset_results) == 1:
    axes = [axes]

fig.suptitle("Part 14: Cross-Asset Model Comparison",
             fontsize=14, fontweight="bold", color="white")

model_keys   = ["naive", "gbm", "logistic", "rf", "lstm", "gru"]
model_labels = ["Naive", "GBM+MC", "Logistic", "RF", "LSTM", "GRU"]
bar_colors   = ["#888888", "#e0c45c", "#5ce0a0",
                "#5cb8e0", "#a05ce0", "#e05c8a"]

for ax, row in zip(axes, all_asset_results):
    ticker = row["ticker"]
    accs   = [row[f"{k}_accuracy"] * 100 for k in model_keys]
    floor  = row["naive_accuracy"] * 100

    bars = ax.bar(model_labels, accs, color=bar_colors,
                  alpha=0.85, edgecolor="white")
    ax.axhline(floor, color="white", linestyle="--",
               linewidth=1.5, label=f"Floor: {floor:.1f}%")
    ax.axhline(SP500_BASELINE * 100, color="yellow", linestyle=":",
               linewidth=1.2, label="S&P 500 floor: 53.9%")
    ax.axhline(50, color="gray", linestyle=":", linewidth=1)
    ax.set_title(ticker, color="white", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", color="white")
    ax.set_ylim(40, 75)
    ax.legend(fontsize=8)
    ax.tick_params(colors="white")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom",
                fontsize=8, color="white")

plt.tight_layout()
cross_fig_path = os.path.join(figures_dir,
                               "part14_cross_asset_comparison.png")
plt.savefig(cross_fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved cross-asset figure: {cross_fig_path}")

print(f"\n{'='*60}")
print("PART 14 COMPLETE")
print(f"{'='*60}")
print("Results saved to:")
for row in all_asset_results:
    t = row["ticker"]
    print(f"  results/{t}/{t}_model_comparison.csv")
print(f"  results/cross_asset_summary.csv")
print("Figures saved to:")
for row in all_asset_results:
    t = row["ticker"]
    print(f"  figures/{t}/{t}_model_comparison.png")
print(f"  figures/part14_cross_asset_comparison.png")