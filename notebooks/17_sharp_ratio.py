# PART 17: SHARPE RATIO SANITY CHECK
# What this part does:
# Simulates a simple trading strategy based on each model's direction
# predictions and computes the Sharpe ratio — a measure of risk-adjusted
# return. Tests whether any model produces better risk-adjusted returns
# than a simple buy-and-hold strategy on S&P 500.
#
# Trading strategy:
#   If model predicts UP (1): go long (buy) — earn the actual return
#   If model predicts DOWN (0): go to cash — earn 0% for that day
#   No transaction costs, no slippage — best-case scenario for models
#
# Sharpe Ratio formula:
#   Sharpe = (mean daily return / std daily return) * sqrt(252)
#   sqrt(252) annualizes the ratio (252 trading days per year)
#   Higher Sharpe = better risk-adjusted return
#   Sharpe > 1.0 is generally considered good
#   Sharpe > 2.0 is considered excellent
#
# Why this matters:
# A model might get 54% direction accuracy but still generate poor
# returns if it misses the biggest up days. Conversely, a model might
# do worse on accuracy but time its predictions better. The Sharpe
# ratio captures the full picture of trading performance.


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"]  = "#0e0e0e"
plt.rcParams["axes.facecolor"]    = "#1a1a1a"
plt.rcParams["savefig.facecolor"] = "#0e0e0e"

script_dir = os.path.dirname(os.path.abspath(__file__))

# STEP 1: LOAD DATA AND PREDICTIONS

features_path = os.path.join(
    script_dir, "..", "data processed", "sp500_features.csv"
)
data = pd.read_csv(features_path, index_col=0, parse_dates=True)
data = data.dropna()

print(f"Data loaded: {data.index[0].date()} to {data.index[-1].date()} "
      f"({len(data)} rows)")

# Load saved prediction arrays from Part 15
preds_dir = os.path.join(script_dir, "..", "results", "predictions")

models_to_load = ["naive", "arima", "logistic", "lstm"]
predictions    = {}
true_labels    = {}

for model in models_to_load:
    path = os.path.join(preds_dir, f"sp500_{model}_preds.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        predictions[model] = df["predictions"].values
        true_labels[model] = df["true"].values
        print(f"Loaded {model}: {len(predictions[model])} predictions")
    else:
        print(f"WARNING: {path} not found. Run Part 15 first.")


# STEP 2: LOAD ACTUAL RETURNS FOR THE SAME PERIOD

# We need the actual S&P 500 returns for the prediction period.
# The prediction arrays from Part 15 cover the walk-forward validation
# windows — we align actual returns to these windows.
#
# Strategy: use the actual next-day return (target_return) for each
# prediction day. If model says UP, we earn that return. If DOWN, we
# earn 0 (sitting in cash).


# Get actual returns aligned to prediction period
# Predictions cover the last portion of the dataset (walk-forward windows)
# We use target_return as the actual daily return earned
actual_returns = data["target_return"].values

# Align to prediction length (same as Part 15 alignment)
n_preds = min(len(v) for v in predictions.values())

# Take the last n_preds returns to match prediction windows
actual_returns_aligned = actual_returns[-n_preds:]

print(f"\nAligned returns: {n_preds} days")
print(f"Return period: approximately "
      f"{data.index[-n_preds].date()} to {data.index[-1].date()}")


# STEP 3: SIMULATE TRADING STRATEGIES


def compute_sharpe(daily_returns, annualize=True):
    """
    Computes Sharpe ratio from daily return series.
    Assumes risk-free rate = 0 for simplicity (standard in academic papers).

    Parameters:
        daily_returns : array of daily portfolio returns
        annualize     : if True, multiplies by sqrt(252)

    Returns:
        sharpe : annualized Sharpe ratio
        mean_r : mean daily return
        std_r  : std of daily returns
        total_r: total cumulative return
    """
    mean_r  = np.mean(daily_returns)
    std_r   = np.std(daily_returns, ddof=1)

    if std_r == 0:
        return 0, mean_r, std_r, 0

    sharpe  = mean_r / std_r
    if annualize:
        sharpe *= np.sqrt(252)

    total_r = np.prod(1 + daily_returns) - 1

    return sharpe, mean_r, std_r, total_r


def simulate_strategy(preds, actual_returns):
    """
    Simulates trading strategy based on model predictions.
    Long if pred=1 (up), cash if pred=0 (down).

    Parameters:
        preds          : binary prediction array (0 or 1)
        actual_returns : actual daily returns for the same period

    Returns:
        strategy_returns : daily returns from the strategy
    """
    # Align lengths
    min_len = min(len(preds), len(actual_returns))
    preds          = preds[-min_len:]
    actual_returns = actual_returns[-min_len:]

    # Strategy: earn return if predicted up, earn 0 if predicted down
    strategy_returns = preds * actual_returns
    return strategy_returns


# --- Buy and Hold benchmark ---
# Simply holding S&P 500 every day — no prediction needed
buyhold_returns = actual_returns_aligned.copy()
bh_sharpe, bh_mean, bh_std, bh_total = compute_sharpe(buyhold_returns)

print(f"\n{'='*60}")
print("TRADING STRATEGY RESULTS")
print(f"{'='*60}")
print(f"\nBuy and Hold (benchmark):")
print(f"  Sharpe Ratio:    {bh_sharpe:.4f}")
print(f"  Mean Daily Ret:  {bh_mean*100:.4f}%")
print(f"  Daily Std:       {bh_std*100:.4f}%")
print(f"  Total Return:    {bh_total*100:.2f}%")

results = []
strategy_returns_dict = {}

for model_name, preds in predictions.items():
    strat_returns = simulate_strategy(preds, actual_returns_aligned)
    sharpe, mean_r, std_r, total_r = compute_sharpe(strat_returns)

    strategy_returns_dict[model_name] = strat_returns

    acc = np.mean(preds[-len(strat_returns):] ==
                  true_labels[model_name][-len(strat_returns):])

    print(f"\n{model_name.capitalize()} Strategy:")
    print(f"  Direction Accuracy: {acc*100:.2f}%")
    print(f"  Sharpe Ratio:       {sharpe:.4f}")
    print(f"  Mean Daily Ret:     {mean_r*100:.4f}%")
    print(f"  Daily Std:          {std_r*100:.4f}%")
    print(f"  Total Return:       {total_r*100:.2f}%")
    print(f"  vs Buy & Hold:      {(sharpe - bh_sharpe):+.4f} Sharpe")

    results.append({
        "model":          model_name,
        "accuracy":       round(acc, 4),
        "sharpe_ratio":   round(sharpe, 4),
        "mean_daily_ret": round(mean_r, 6),
        "daily_std":      round(std_r, 6),
        "total_return":   round(total_r, 4),
        "vs_buyhold":     round(sharpe - bh_sharpe, 4),
    })

# Add buy and hold to results
results.insert(0, {
    "model":          "buy_and_hold",
    "accuracy":       None,
    "sharpe_ratio":   round(bh_sharpe, 4),
    "mean_daily_ret": round(bh_mean, 6),
    "daily_std":      round(bh_std, 6),
    "total_return":   round(bh_total, 4),
    "vs_buyhold":     0.0,
})


# STEP 4: SAVE RESULTS


results_dir = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_dir, "sharpe_ratio_results.csv"),
                  index=False)
print(f"\nSaved results to results/sharpe_ratio_results.csv")


# STEP 5: VISUALIZATIONS


figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Part 17: Sharpe Ratio Sanity Check",
             fontsize=14, fontweight="bold", color="white")

# --- Plot 1: Sharpe Ratios ---
ax1 = axes[0]
model_names  = ["Buy &\nHold"] + [m.capitalize() for m in predictions.keys()]
sharpe_vals  = [bh_sharpe] + [r["sharpe_ratio"] for r in results[1:]]
bar_colors   = ["#e0c45c"] + ["#888888", "#e0a85c", "#5ce0a0", "#a05ce0"]

bars = ax1.bar(model_names, sharpe_vals, color=bar_colors,
               alpha=0.85, edgecolor="white")
ax1.axhline(bh_sharpe, color="yellow", linestyle="--", linewidth=1.5,
            label=f"Buy & Hold: {bh_sharpe:.3f}")
ax1.axhline(0, color="gray", linestyle=":", linewidth=1)
ax1.set_ylabel("Annualized Sharpe Ratio", color="white")
ax1.set_title("Sharpe Ratio by Strategy", color="white")
ax1.legend(fontsize=9)
ax1.tick_params(colors="white")

for bar, val in zip(bars, sharpe_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=9, color="white")

# --- Plot 2: Cumulative Returns ---
ax2 = axes[1]
colors_cum = ["#e0c45c", "#888888", "#e0a85c", "#5ce0a0", "#a05ce0"]
labels_cum = ["Buy & Hold", "Naive", "ARIMA", "Logistic", "LSTM"]

# Buy and hold cumulative
cum_bh = np.cumprod(1 + buyhold_returns) - 1
ax2.plot(cum_bh * 100, color="#e0c45c", linewidth=1.5,
         label="Buy & Hold")

for (model_name, strat_ret), color, label in zip(
        strategy_returns_dict.items(), colors_cum[1:], labels_cum[1:]):
    cum = np.cumprod(1 + strat_ret) - 1
    ax2.plot(cum * 100, color=color, linewidth=1.0,
             alpha=0.8, label=label)

ax2.axhline(0, color="gray", linestyle=":", linewidth=1)
ax2.set_xlabel("Trading Day", color="white")
ax2.set_ylabel("Cumulative Return (%)", color="white")
ax2.set_title("Cumulative Returns by Strategy", color="white")
ax2.legend(fontsize=8)
ax2.tick_params(colors="white")

# --- Plot 3: Total Return comparison ---
ax3 = axes[2]
total_rets  = [bh_total] + [r["total_return"] for r in results[1:]]
bar_colors3 = ["#e0c45c"] + ["#888888", "#e0a85c", "#5ce0a0", "#a05ce0"]

bars3 = ax3.bar(model_names, [r * 100 for r in total_rets],
                color=bar_colors3, alpha=0.85, edgecolor="white")
ax3.axhline(bh_total * 100, color="yellow", linestyle="--",
            linewidth=1.5, label=f"Buy & Hold: {bh_total*100:.1f}%")
ax3.axhline(0, color="gray", linestyle=":", linewidth=1)
ax3.set_ylabel("Total Return (%)", color="white")
ax3.set_title("Total Return by Strategy", color="white")
ax3.legend(fontsize=9)
ax3.tick_params(colors="white")

for bar, val in zip(bars3, total_rets):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f"{val*100:.1f}%", ha="center", va="bottom",
             fontsize=9, color="white")

plt.tight_layout()
fig_path = os.path.join(figures_dir, "part17_sharpe_ratio.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to: {fig_path}")


# STEP 6: FINAL SUMMARY


print("\n" + "=" * 60)
print("SHARPE RATIO SUMMARY")
print("=" * 60)
print(f"\n{'Strategy':<15} {'Sharpe':>10} {'Total Ret':>12} "
      f"{'vs B&H':>10}")
print("-" * 50)
print(f"{'Buy & Hold':<15} {bh_sharpe:>10.4f} "
      f"{bh_total*100:>11.2f}% {'—':>10}")

for r in results[1:]:
    print(f"{r['model'].capitalize():<15} {r['sharpe_ratio']:>10.4f} "
          f"{r['total_return']*100:>11.2f}% "
          f"{r['vs_buyhold']:>+10.4f}")

best_model = max(results[1:], key=lambda x: x["sharpe_ratio"])
print(f"\nBest model Sharpe: {best_model['model'].capitalize()} "
      f"({best_model['sharpe_ratio']:.4f})")
print(f"Buy & Hold Sharpe: {bh_sharpe:.4f}")

if best_model["sharpe_ratio"] > bh_sharpe:
    print(f"\nBest model BEATS Buy & Hold by "
          f"{best_model['sharpe_ratio'] - bh_sharpe:+.4f} Sharpe points.")
    print("Note: no transaction costs assumed — real trading would reduce this.")
else:
    print(f"\nNo model beats Buy & Hold on risk-adjusted returns.")
    print("Consistent with accuracy findings — models add no practical value.")