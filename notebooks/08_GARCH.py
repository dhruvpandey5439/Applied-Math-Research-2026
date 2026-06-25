# =============================================================================
# PART 8: GARCH(1,1) MODEL
# =============================================================================
# Research Question:
# To what extent do increasingly complex machine learning models improve
# predictive performance compared to classical statistical AND mathematical
# methods when forecasting financial time-series data?
#
# What GARCH does:
# GARCH models time-varying volatility -- how "jumpy" the market is expected
# to be tomorrow, based on how jumpy it has been recently. It does NOT predict
# direction. Instead it answers: "how uncertain is tomorrow?"
#
# The equation:
#   sigma^2(t) = omega + alpha * epsilon^2(t-1) + beta * sigma^2(t-1)
#
#   sigma^2(t)       = today's variance (what we're predicting)
#   omega            = long-run average variance (a constant baseline)
#   epsilon^2(t-1)   = yesterday's squared shock (surprise move)
#   sigma^2(t-1)     = yesterday's variance (how volatile it already was)
#
# GARCH(1,1) means: one lag of the shock term (alpha) and one lag of the
# variance term (beta). This is the standard specification used in virtually
# all financial volatility research since Bollerslev (1986).
#
# How GARCH contributes to this project:
# Since it predicts volatility (not direction), we use it in two ways:
#   1. As a standalone volatility forecast (RMSE on predicted vs actual vol)
#   2. As a signal fed into a direction rule:
#      -- High predicted volatility → predict "down" (risk-off signal)
#      -- Low predicted volatility  → predict "up"  (calm market signal)
#      This lets us compute direction accuracy for fair comparison.
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = "#0e0e0e"
plt.rcParams["axes.facecolor"]   = "#1a1a1a"
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
print(f"Columns: {list(data.columns)}\n")

returns = data["Return"] * 100

target_direction = data["target_direction"]
target_return    = data["target_return"]

# =============================================================================
# STEP 2: WALK-FORWARD VALIDATION SETUP
# =============================================================================

n           = len(returns)
n_splits    = 5
test_size   = int(n * 0.10)
min_train   = int(n * 0.50)

folds = []
for i in range(n_splits):
    test_end   = n - (n_splits - 1 - i) * test_size
    test_start = test_end - test_size
    train_end  = test_start
    if train_end < min_train:
        continue
    folds.append((0, train_end, test_start, test_end))

print(f"Walk-forward validation: {len(folds)} folds\n")

# =============================================================================
# STEP 3: GARCH(1,1) WALK-FORWARD LOOP
# =============================================================================

fold_results = []

for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):

    train_returns = returns.iloc[train_start:train_end]
    test_returns  = returns.iloc[test_start:test_end]
    test_dir      = target_direction.iloc[test_start:test_end]

    print(f"Fold {fold_idx + 1}:")
    print(f"  Train: {returns.index[train_start].date()} → "
          f"{returns.index[train_end - 1].date()} ({train_end - train_start} days)")
    print(f"  Test:  {returns.index[test_start].date()} → "
          f"{returns.index[test_end - 1].date()} ({test_end - test_start} days)")

    predicted_vol  = []
    realized_vol   = []
    predicted_dirs = []

    train_realized_vol = train_returns.rolling(window=5).std().dropna()
    vol_threshold = train_realized_vol.median()

    for step in range(len(test_returns)):

        current_train = pd.concat([
            train_returns,
            test_returns.iloc[:step]
        ])

        try:
            model = arch_model(
                current_train,
                mean='Zero',
                vol='GARCH',
                p=1,
                q=1,
                rescale=False
            )
            result = model.fit(disp='off', show_warning=False)
            forecast = result.forecast(horizon=1)
            pred_variance = forecast.variance.values[-1, 0]
            pred_vol      = np.sqrt(pred_variance)

        except Exception:
            pred_vol = current_train.tail(20).std()

        if step >= 4:
            realized = test_returns.iloc[step - 4: step + 1].std()
        else:
            combined = pd.concat([train_returns.tail(5 - step - 1),
                                   test_returns.iloc[:step + 1]])
            realized = combined.std()

        predicted_vol.append(pred_vol)
        realized_vol.append(realized)

        if pred_vol > vol_threshold:
            predicted_dirs.append(0)
        else:
            predicted_dirs.append(1)

    predicted_vol  = np.array(predicted_vol)
    realized_vol   = np.array(realized_vol)
    predicted_dirs = np.array(predicted_dirs)
    actual_dirs    = test_dir.values

    vol_rmse = np.sqrt(np.mean((predicted_vol - realized_vol) ** 2))
    vol_mae  = np.mean(np.abs(predicted_vol - realized_vol))
    dir_acc  = accuracy_score(actual_dirs, predicted_dirs)
    dir_f1   = f1_score(actual_dirs, predicted_dirs, zero_division=0)

    print(f"  Vol RMSE: {vol_rmse:.4f}%  |  Vol MAE: {vol_mae:.4f}%")
    print(f"  Direction Accuracy: {dir_acc:.4f} ({dir_acc*100:.2f}%)")
    print(f"  F1 Score: {dir_f1:.4f}\n")

    fold_results.append({
        "fold":         fold_idx + 1,
        "train_start":  returns.index[train_start].date(),
        "train_end":    returns.index[train_end - 1].date(),
        "test_start":   returns.index[test_start].date(),
        "test_end":     returns.index[test_end - 1].date(),
        "vol_rmse":     vol_rmse,
        "vol_mae":      vol_mae,
        "dir_accuracy": dir_acc,
        "f1_score":     dir_f1,
        "pred_vol":     predicted_vol,
        "realized_vol": realized_vol,
        "pred_dirs":    predicted_dirs,
        "actual_dirs":  actual_dirs,
    })

# =============================================================================
# STEP 4: AGGREGATE RESULTS
# =============================================================================

results_df = pd.DataFrame([{
    "fold":         r["fold"],
    "vol_rmse":     r["vol_rmse"],
    "vol_mae":      r["vol_mae"],
    "dir_accuracy": r["dir_accuracy"],
    "f1_score":     r["f1_score"],
} for r in fold_results])

mean_vol_rmse  = results_df["vol_rmse"].mean()
mean_vol_mae   = results_df["vol_mae"].mean()
mean_dir_acc   = results_df["dir_accuracy"].mean()
mean_f1        = results_df["f1_score"].mean()

naive_baseline = 0.5393

print("=" * 60)
print("GARCH(1,1) SUMMARY ACROSS ALL FOLDS")
print("=" * 60)
print(f"Mean Volatility RMSE:     {mean_vol_rmse:.6f}%")
print(f"Mean Volatility MAE:      {mean_vol_mae:.6f}%")
print(f"Mean Direction Accuracy:  {mean_dir_acc:.4f} ({mean_dir_acc*100:.2f}%)")
print(f"Mean F1 Score:            {mean_f1:.4f}")
print(f"vs Naive Baseline:        {(mean_dir_acc - naive_baseline)*100:+.2f} pp")
print("=" * 60)

# =============================================================================
# STEP 5: SAVE RESULTS TO model_comparison.csv
# =============================================================================

results_dir = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)
comparison_path = os.path.join(results_dir, "model_comparison.csv")

new_row = pd.DataFrame([{
    "model":       "GARCH(1,1)",
    "accuracy":    round(mean_dir_acc, 4),
    "f1_score":    round(mean_f1, 4),
    "rmse":        round(mean_vol_rmse, 6),
    "mae":         round(mean_vol_mae, 6),
    "test_start":  fold_results[0]["test_start"],
    "test_end":    fold_results[-1]["test_end"],
    "notes":       (
        "Volatility model. Direction predicted via vol threshold rule "
        "(high vol → down, low vol → up). RMSE/MAE are on % volatility, "
        "not return levels."
    ),
}])

if os.path.exists(comparison_path):
    existing = pd.read_csv(comparison_path)
    existing = existing[existing["model"] != "GARCH(1,1)"]
    combined = pd.concat([existing, new_row], ignore_index=True)
else:
    combined = new_row

combined.to_csv(comparison_path, index=False)
print(f"\nSaved results to: {comparison_path}")

# =============================================================================
# STEP 6: VISUALIZATIONS
# =============================================================================

figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Part 8: GARCH(1,1) Results", fontsize=14, fontweight="bold", color="white")

# --- Plot 1: Direction Accuracy per Fold ---
ax1       = axes[0]
fold_nums = results_df["fold"].tolist()
dir_accs  = results_df["dir_accuracy"].tolist()
colors    = ["#e05c5c" if a < naive_baseline else "#5ce0a0" for a in dir_accs]
bars      = ax1.bar(fold_nums, [a * 100 for a in dir_accs], color=colors, alpha=0.85)
ax1.axhline(naive_baseline * 100, color="white", linestyle="--",
            linewidth=1.5, label=f"Naive baseline ({naive_baseline*100:.1f}%)")
ax1.axhline(50, color="gray", linestyle=":", linewidth=1, label="50% random")
ax1.set_xlabel("Fold", color="white")
ax1.set_ylabel("Direction Accuracy (%)", color="white")
ax1.set_title("Direction Accuracy per Fold", color="white")
ax1.set_xticks(fold_nums)
ax1.legend(fontsize=8)
ax1.set_ylim(40, 65)
ax1.tick_params(colors="white")
for bar, acc in zip(bars, dir_accs):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=8, color="white")

# --- Plot 2: Predicted vs Realized Volatility (last fold) ---
ax2  = axes[1]
last = fold_results[-1]
x    = range(len(last["pred_vol"]))
ax2.plot(x, last["realized_vol"], color="#aaaaaa", linewidth=1,
         label="Realized vol", alpha=0.8)
ax2.plot(x, last["pred_vol"], color="#5cb8e0", linewidth=1.2,
         label="Predicted vol (GARCH)", alpha=0.9)
ax2.set_xlabel("Test Day (last fold)", color="white")
ax2.set_ylabel("Volatility (%)", color="white")
ax2.set_title("Predicted vs Realized Volatility\n(Last Fold)", color="white")
ax2.legend(fontsize=8)
ax2.tick_params(colors="white")

# --- Plot 3: Model Comparison Bar Chart ---
ax3        = axes[2]
models     = ["Always Up\n(Naive)", "Persistence\n(Naive)", "ARIMA\n(1,0,1)", "GARCH\n(1,1)"]
accuracies = [53.93, 49.77, 50.73, mean_dir_acc * 100]
bar_colors = ["#888888", "#888888", "#e0a85c", "#5cb8e0"]
bars3      = ax3.bar(models, accuracies, color=bar_colors, alpha=0.85)
ax3.axhline(naive_baseline * 100, color="white", linestyle="--",
            linewidth=1.5, label=f"Naive floor ({naive_baseline*100:.1f}%)")
ax3.axhline(50, color="gray", linestyle=":", linewidth=1)
ax3.set_ylabel("Direction Accuracy (%)", color="white")
ax3.set_title("All Models So Far", color="white")
ax3.set_ylim(40, 65)
ax3.legend(fontsize=8)
ax3.tick_params(colors="white")
for bar, acc in zip(bars3, accuracies):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{acc:.1f}%", ha="center", va="bottom", fontsize=8, color="white")

plt.tight_layout()
fig_path = os.path.join(figures_dir, "part8_garch_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to: {fig_path}")

# =============================================================================
# STEP 7: FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Model:                   GARCH(1,1)")
print(f"Primary purpose:         Volatility forecasting")
print(f"Direction method:        High vol → down, Low vol → up")
print(f"Mean direction accuracy: {mean_dir_acc*100:.2f}%")
print(f"Mean F1 score:           {mean_f1:.4f}")
print(f"vs Naive baseline:       {(mean_dir_acc - naive_baseline)*100:+.2f} pp")
print(f"Mean vol RMSE:           {mean_vol_rmse:.4f}%")
print(f"Mean vol MAE:            {mean_vol_mae:.4f}%")
print("=" * 60)
print("\nPart 8 complete. Next: Part 9 -- GBM + Monte Carlo")