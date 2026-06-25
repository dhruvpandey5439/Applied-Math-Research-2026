# =============================================================================
# PART 9: GBM + MONTE CARLO SIMULATION
# =============================================================================
# Research Question:
# To what extent do increasingly complex machine learning models improve
# predictive performance compared to classical statistical AND mathematical
# methods when forecasting financial time-series data?
#
# What GBM is:
# Geometric Brownian Motion is the mathematical model of how stock prices move.
# It is the foundation of the entire mathematical finance tradition --
# Black-Scholes, options pricing, and modern risk management all sit on top of it.
#
# The equation (stochastic differential equation):
#   dS = μS dt + σS dW
#
#   S   = current price
#   μ   = drift (average daily upward tendency of the market)
#   σ   = volatility (how much prices jump around)
#   dt  = one time step (one day here)
#   dW  = random noise drawn from a normal distribution
#
# In plain English: tomorrow's price = today's price x (drift + randomness)
#
# What Monte Carlo means:
# Instead of solving the equation once, we simulate it N=1000 times,
# each time drawing DIFFERENT random noise. This gives us 1,000 possible
# futures. We take the MEDIAN simulated path as our direction forecast.
#
# Why this is different from ARIMA and GARCH:
# ARIMA and GARCH learn patterns from past data.
# GBM does NOT learn -- it says "prices follow drift + randomness"
# and simulates from that assumption. The only things calibrated from
# data are μ (mean daily return) and σ (std of daily returns).
# Everything else is pure mathematical theory.
#
# How GBM contributes to this project:
# GBM represents the MATHEMATICAL FINANCE tradition -- the third category
# in our three-way comparison. This is the core novelty of the project.
# Most papers only compare statistical vs ML. We explicitly include
# the mathematical tradition as a third benchmark.
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

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
print(f"Columns: {list(data.columns)}\n")

returns          = data["Return"]
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
# STEP 3: MONTE CARLO GBM SIMULATION FUNCTION
# =============================================================================

def simulate_gbm_direction(mu, sigma, S0, n_simulations=1000, dt=1):
    """
    Simulates N price paths one step forward using GBM.

    Parameters:
        mu            : daily drift (mean of log returns on training data)
        sigma         : daily volatility (std of log returns on training data)
        S0            : today's price (starting point for all simulations)
        n_simulations : number of random paths to simulate (default 1000)
        dt            : time step in days (1 = one day ahead)

    Returns:
        direction        : 1 if median simulated price > S0, else 0
        simulated_prices : array of all simulated next-day prices

    How it works:
        The discrete GBM formula for one step is:
        S(t+1) = S(t) * exp((μ - 0.5*σ²)*dt + σ*sqrt(dt)*Z)

        Where Z ~ N(0,1) is standard normal random noise.
        The (μ - 0.5*σ²) term is the Ito correction -- without it,
        the expected log return would be biased upward.
        We draw 1000 different Z values → 1000 different S(t+1) values.
        The median of those 1000 values is our point forecast.
    """
    Z                = np.random.standard_normal(n_simulations)
    simulated_prices = S0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    median_price     = np.median(simulated_prices)
    direction        = 1 if median_price > S0 else 0
    return direction, simulated_prices

# =============================================================================
# STEP 4: GBM WALK-FORWARD LOOP
# =============================================================================

np.random.seed(42)

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

    mu_train    = train_returns.mean()
    sigma_train = train_returns.std()

    print(f"  Calibrated μ (drift):     {mu_train:.6f}")
    print(f"  Calibrated σ (vol):       {sigma_train:.6f}")

    predicted_dirs    = []
    predicted_returns = []

    for step in range(len(test_returns)):

        today_idx = test_start + step
        S0        = data["Close"].iloc[today_idx]

        current_returns = pd.concat([
            train_returns,
            test_returns.iloc[:step]
        ])
        mu    = current_returns.mean()
        sigma = current_returns.std()

        direction, simulated_prices = simulate_gbm_direction(
            mu=mu,
            sigma=sigma,
            S0=S0,
            n_simulations=1000,
            dt=1
        )

        median_price     = np.median(simulated_prices)
        predicted_return = (median_price - S0) / S0

        predicted_dirs.append(direction)
        predicted_returns.append(predicted_return)

    predicted_dirs    = np.array(predicted_dirs)
    predicted_returns = np.array(predicted_returns)
    actual_dirs       = test_dir.values
    actual_returns    = target_return.iloc[test_start:test_end].values

    dir_acc  = accuracy_score(actual_dirs, predicted_dirs)
    dir_f1   = f1_score(actual_dirs, predicted_dirs, zero_division=0)
    ret_rmse = np.sqrt(np.mean((predicted_returns - actual_returns) ** 2))
    ret_mae  = np.mean(np.abs(predicted_returns - actual_returns))

    print(f"  Direction Accuracy: {dir_acc:.4f} ({dir_acc*100:.2f}%)")
    print(f"  F1 Score:           {dir_f1:.4f}")
    print(f"  Return RMSE:        {ret_rmse:.6f}")
    print(f"  Return MAE:         {ret_mae:.6f}\n")

    fold_results.append({
        "fold":              fold_idx + 1,
        "train_start":       returns.index[train_start].date(),
        "train_end":         returns.index[train_end - 1].date(),
        "test_start":        returns.index[test_start].date(),
        "test_end":          returns.index[test_end - 1].date(),
        "dir_accuracy":      dir_acc,
        "f1_score":          dir_f1,
        "ret_rmse":          ret_rmse,
        "ret_mae":           ret_mae,
        "predicted_dirs":    predicted_dirs,
        "actual_dirs":       actual_dirs,
        "predicted_returns": predicted_returns,
        "actual_returns":    actual_returns,
    })

# =============================================================================
# STEP 5: AGGREGATE RESULTS
# =============================================================================

results_df = pd.DataFrame([{
    "fold":         r["fold"],
    "dir_accuracy": r["dir_accuracy"],
    "f1_score":     r["f1_score"],
    "ret_rmse":     r["ret_rmse"],
    "ret_mae":      r["ret_mae"],
} for r in fold_results])

mean_dir_acc  = results_df["dir_accuracy"].mean()
mean_f1       = results_df["f1_score"].mean()
mean_ret_rmse = results_df["ret_rmse"].mean()
mean_ret_mae  = results_df["ret_mae"].mean()

naive_baseline = 0.5393

print("=" * 60)
print("GBM + MONTE CARLO SUMMARY ACROSS ALL FOLDS")
print("=" * 60)
print(f"Mean Direction Accuracy:  {mean_dir_acc:.4f} ({mean_dir_acc*100:.2f}%)")
print(f"Mean F1 Score:            {mean_f1:.4f}")
print(f"Mean Return RMSE:         {mean_ret_rmse:.6f}")
print(f"Mean Return MAE:          {mean_ret_mae:.6f}")
print(f"vs Naive Baseline:        {(mean_dir_acc - naive_baseline)*100:+.2f} pp")
print("=" * 60)

# =============================================================================
# STEP 6: SAVE RESULTS TO model_comparison.csv
# =============================================================================

results_dir     = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)
comparison_path = os.path.join(results_dir, "model_comparison.csv")

new_row = pd.DataFrame([{
    "model":      "GBM + Monte Carlo",
    "accuracy":   round(mean_dir_acc, 4),
    "f1_score":   round(mean_f1, 4),
    "rmse":       round(mean_ret_rmse, 6),
    "mae":        round(mean_ret_mae, 6),
    "test_start": fold_results[0]["test_start"],
    "test_end":   fold_results[-1]["test_end"],
    "notes": (
        "Mathematical finance tradition. GBM simulates 1000 price paths "
        "per day using calibrated mu and sigma. Median path used for "
        "direction forecast. No pattern learning -- pure stochastic simulation."
    ),
}])

if os.path.exists(comparison_path):
    existing = pd.read_csv(comparison_path)
    existing = existing[existing["model"] != "GBM + Monte Carlo"]
    combined = pd.concat([existing, new_row], ignore_index=True)
else:
    combined = new_row

combined.to_csv(comparison_path, index=False)
print(f"\nSaved results to: {comparison_path}")

# =============================================================================
# STEP 7: VISUALIZATIONS
# =============================================================================

figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Part 9: GBM + Monte Carlo Results", fontsize=14, fontweight="bold", color="white")

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

# --- Plot 2: Monte Carlo Distribution (last fold, last test day) ---
ax2          = axes[1]
last_returns = returns.iloc[folds[-1][0]:folds[-1][1]]
mu_viz       = last_returns.mean()
sigma_viz    = last_returns.std()
last_S0      = data["Close"].iloc[folds[-1][3] - 1]

np.random.seed(99)
n_viz          = 200
Z_viz          = np.random.standard_normal(n_viz)
sim_prices_viz = last_S0 * np.exp(
    (mu_viz - 0.5 * sigma_viz**2) + sigma_viz * Z_viz
)

ax2.hist(sim_prices_viz, bins=30, color="#5cb8e0", alpha=0.7, edgecolor="none")
ax2.axvline(last_S0, color="white", linewidth=1.5, linestyle="--",
            label=f"Today: {last_S0:.1f}")
ax2.axvline(np.median(sim_prices_viz), color="#5ce0a0", linewidth=1.5,
            label=f"Median: {np.median(sim_prices_viz):.1f}")
ax2.set_xlabel("Simulated Next-Day Price", color="white")
ax2.set_ylabel("Count", color="white")
ax2.set_title("Monte Carlo Distribution\n(200 paths, last test day)", color="white")
ax2.legend(fontsize=8)
ax2.tick_params(colors="white")

# --- Plot 3: All Models Comparison ---
ax3        = axes[2]
models     = ["Always Up\n(Naive)", "Persistence\n(Naive)", "ARIMA\n(1,0,1)",
              "GARCH\n(1,1)", "GBM+\nMonte Carlo"]
accuracies = [53.93, 49.77, 50.73, 49.23, mean_dir_acc * 100]
bar_colors = ["#888888", "#888888", "#e0a85c", "#5cb8e0", "#a05ce0"]
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
fig_path = os.path.join(figures_dir, "part9_gbm_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to: {fig_path}")

# =============================================================================
# STEP 8: FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Model:                   GBM + Monte Carlo")
print(f"Tradition:               Mathematical Finance")
print(f"Simulation paths:        1000 per day")
print(f"Direction method:        Median simulated price vs today price")
print(f"Mean direction accuracy: {mean_dir_acc*100:.2f}%")
print(f"Mean F1 score:           {mean_f1:.4f}")
print(f"Mean return RMSE:        {mean_ret_rmse:.6f}")
print(f"Mean return MAE:         {mean_ret_mae:.6f}")
print(f"vs Naive baseline:       {(mean_dir_acc - naive_baseline)*100:+.2f} pp")
print("=" * 60)
