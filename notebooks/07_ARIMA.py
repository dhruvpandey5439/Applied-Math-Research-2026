import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_squared_error, mean_absolute_error)
from statsmodels.tsa.arima.model import ARIMA
 
warnings.filterwarnings("ignore")  # ARIMA produces harmless convergence warnings
 
# =============================================================================
# STEP 1: Load data
# =============================================================================
script_dir    = os.path.dirname(os.path.abspath(__file__))
features_path = os.path.join(script_dir, "..", "data processed", "sp500_features.csv")
 
data = pd.read_csv(features_path, index_col=0, parse_dates=True)
print("Loaded data shape:", data.shape)
print("Date range:", data.index.min().date(), "to", data.index.max().date())
 
# We only need the Return series for ARIMA -- it models the series itself,
# not the engineered features we built in Day 3. ARIMA is a univariate model:
# it only looks at one variable (past returns) to predict future returns.
returns = data["Return"]
plt.style.use("dark_background")

# =============================================================================
# STEP 2: Choose ARIMA order (p, d, q)
# =============================================================================
# p = number of AR lags (how many past returns to use)
# d = differencing order (0 since returns are already stationary)
# q = number of MA lags (how many past errors to use)
#
# From Day 2: ACF/PACF on returns showed almost no significant lags.
# This suggests small p and q values. We use ARIMA(1,0,1) as a standard
# starting point -- one AR lag and one MA lag. This is the most common
# specification in the financial forecasting literature for daily returns.
#
# In a more advanced study, you'd use AIC/BIC criteria to select the
# optimal order automatically. For this project, ARIMA(1,0,1) is the
# appropriate benchmark choice, consistent with the literature.
 
p, d, q = 1, 0, 1
print(f"\nARIMA order: ({p}, {d}, {q})")
print("p=1: use yesterday's return as AR input")
print("d=0: no differencing needed (returns already stationary)")
print("q=1: use yesterday's forecast error as MA input")
 
# =============================================================================
# STEP 3: Walk-forward validation with 5 folds
# =============================================================================
# For each fold: fit ARIMA on training data, predict ONE step ahead on
# each day of the test period (re-fitting each time to avoid lookahead).
# This is computationally slower than a single fit but far more honest.
#
# Note: full re-fitting on every single test day would take hours for
# 26 years of data. Instead we fit once per fold on the training window
# and forecast the entire test window -- a standard compromise in the
# literature called "fixed-window" forecasting.
 
tscv = TimeSeriesSplit(n_splits=5)
 
fold_results = []  # store metrics from each fold
 
print("\n=== Walk-Forward Validation: 5 Folds ===")
 
for fold, (train_idx, test_idx) in enumerate(tscv.split(data), start=1):
 
    train_returns = returns.iloc[train_idx]
    test_returns  = returns.iloc[test_idx]
    test_data     = data.iloc[test_idx]
 
    # Fit ARIMA on training data
    model  = ARIMA(train_returns, order=(p, d, q))
    fitted = model.fit()
 
    # Forecast the test period (n_periods steps ahead)
    n_test     = len(test_returns)
    forecast   = fitted.forecast(steps=n_test)
 
    # --- Regression metrics (how close are the predicted return VALUES?) ---
    rmse = np.sqrt(mean_squared_error(test_returns, forecast))
    mae  = mean_absolute_error(test_returns, forecast)
 
    # --- Direction metrics (did we predict UP vs DOWN correctly?) ---
    # Convert predicted return to direction: positive forecast = predict up (1)
    predicted_direction = (forecast > 0).astype(int).values
    actual_direction    = (test_returns > 0).astype(int).values
 
    acc = accuracy_score(actual_direction, predicted_direction)
    f1  = f1_score(actual_direction, predicted_direction, zero_division=0)
 
    fold_results.append({
        "fold":       fold,
        "train_size": len(train_returns),
        "test_size":  len(test_returns),
        "test_start": test_returns.index.min().date(),
        "test_end":   test_returns.index.max().date(),
        "rmse":       rmse,
        "mae":        mae,
        "accuracy":   acc,
        "f1_score":   f1
    })
 
    print(f"\nFold {fold}: train={len(train_returns)} rows | "
          f"test={test_returns.index.min().date()} to {test_returns.index.max().date()}")
    print(f"  RMSE: {rmse:.6f} | MAE: {mae:.6f}")
    print(f"  Direction Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")
 
# =============================================================================
# STEP 4: Summarize across folds
# =============================================================================
results_df = pd.DataFrame(fold_results)
 
mean_rmse     = results_df["rmse"].mean()
mean_mae      = results_df["mae"].mean()
mean_accuracy = results_df["accuracy"].mean()
mean_f1       = results_df["f1_score"].mean()
 
print("\n=== ARIMA Results Summary (mean across 5 folds) ===")
print(f"Mean RMSE:              {mean_rmse:.6f}")
print(f"Mean MAE:               {mean_mae:.6f}")
print(f"Mean Direction Accuracy:{mean_accuracy*100:.2f}%")
print(f"Mean F1 Score:          {mean_f1:.4f}")
print(f"\nNaive baseline (Always Up): 53.93% accuracy")
print(f"ARIMA vs baseline:          {(mean_accuracy - 0.5393)*100:+.2f} percentage points")
 
# =============================================================================
# STEP 5: Append ARIMA results to the model comparison CSV
# =============================================================================
results_dir  = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "model_comparison.csv")
 
# Load existing comparison table (from Day 6) and append ARIMA row
existing = pd.read_csv(results_path)
 
arima_row = pd.DataFrame([{
    "model":      f"ARIMA({p},{d},{q}) -- 5-fold walk-forward",
    "accuracy":   round(mean_accuracy, 4),
    "f1_score":   round(mean_f1, 4),
    "test_start": results_df["test_start"].iloc[0],
    "test_end":   results_df["test_end"].iloc[-1],
    "notes":      f"RMSE={mean_rmse:.6f}, MAE={mean_mae:.6f}. "
                  f"Classical statistical benchmark."
}])
 
updated = pd.concat([existing, arima_row], ignore_index=True)
updated.to_csv(results_path, index=False)
print(f"\nUpdated model comparison table saved to {results_path}")
print(updated.to_string(index=False))
 
# =============================================================================
# STEP 6: Visualize -- fold-by-fold accuracy + predicted vs actual returns
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
 
# Top: accuracy per fold vs naive baseline
axes[0].plot(results_df["fold"], results_df["accuracy"] * 100,
             marker="o", color="steelblue", label="ARIMA accuracy per fold")
axes[0].axhline(y=53.93, color="red", linestyle="--",
                label="Naive baseline (Always Up) 53.93%")
axes[0].axhline(y=50, color="gray", linestyle=":",
                label="50% random chance")
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("Direction Accuracy (%)")
axes[0].set_title("ARIMA Direction Accuracy per Fold vs Naive Baseline")
axes[0].set_ylim(44, 62)
axes[0].legend()
axes[0].grid(alpha=0.3)
 
for _, row in results_df.iterrows():
    axes[0].text(row["fold"], row["accuracy"] * 100 + 0.4,
                 f"{row['accuracy']*100:.1f}%", ha="center", fontsize=9)
 
# Bottom: bar chart of mean metrics
metrics = ["RMSE", "MAE"]
values  = [mean_rmse, mean_mae]
axes[1].bar(metrics, values, color=["steelblue", "darkorange"], width=0.4)
axes[1].set_ylabel("Error (return units)")
axes[1].set_title("ARIMA Mean Regression Error (5-fold average)")
axes[1].grid(axis="y", alpha=0.3)
for i, v in enumerate(values):
    axes[1].text(i, v + 0.0001, f"{v:.6f}", ha="center", fontsize=10)
 
plt.tight_layout()
figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)
fig_path = os.path.join(figures_dir, "day7_arima_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to {fig_path}")
 
# =============================================================================
# STEP 7: Research log summary
# =============================================================================
print("\n" + "="*60)
print("DAY 7 SUMMARY -- add this to your research log:")
print("="*60)
print(f"""
What I did:
  Fitted ARIMA({p},{d},{q}) on S&P 500 daily returns using 5-fold
  walk-forward validation. Evaluated on both regression metrics
  (RMSE, MAE) and direction metrics (accuracy, F1).
 
Results (mean across 5 folds):
  RMSE:               {mean_rmse:.6f}
  MAE:                {mean_mae:.6f}
  Direction Accuracy: {mean_accuracy*100:.2f}%
  F1 Score:           {mean_f1:.4f}
  vs Naive Baseline:  {(mean_accuracy - 0.5393)*100:+.2f} percentage points
 
What this means:
  ARIMA is the classical statistical benchmark. Its direction accuracy
  relative to the 53.93% naive baseline tells us how much predictive
  value past return patterns contain under a linear model. Given the
  near-zero ACF from Day 2, we expected ARIMA to show limited improvement
  over the naive baseline -- consistent with the EMH.
 
Saved:
  results/model_comparison.csv (updated with ARIMA row)
  figures/day7_arima_results.png
""")
 