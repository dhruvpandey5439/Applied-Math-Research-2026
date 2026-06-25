import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# =============================================================================
# STEP 1: Load the feature dataset from Phase 1
# =============================================================================
# This is the file we saved at the end of Day 3 (sp500_features.csv).
# It has all our engineered features AND our two targets:
#   - target_direction: 1 if tomorrow is up, 0 if down
#   - target_return: tomorrow's actual return value

script_dir = os.path.dirname(os.path.abspath(__file__))

# sp500_features.csv lives in AppliedMath_Dev/data processed/
# The script lives in AppliedMath_Dev/notebooks/
# So we go up one level (..) then into "data processed"
features_path = os.path.join(script_dir, "..", "data processed", "sp500_features.csv")

data = pd.read_csv(features_path, index_col=0, parse_dates=True)

print("Loaded data shape:", data.shape)
print("Date range:", data.index.min().date(), "to", data.index.max().date())
print("\nColumns:", list(data.columns))

# Quick sanity check: confirm our two targets exist
assert "target_direction" in data.columns, "Missing target_direction column!"
assert "target_return" in data.columns, "Missing target_return column!"
assert "Return" in data.columns, "Missing Return column!"

# =============================================================================
# STEP 2: Chronological 80/20 train/test split (same as Day 4)
# =============================================================================
split_index = int(len(data) * 0.8)
train = data.iloc[:split_index]
test  = data.iloc[split_index:]

print(f"\nTrain: {train.index.min().date()} to {train.index.max().date()} ({len(train)} rows)")
print(f"Test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} rows)")

y_true = test["target_direction"]
plt.style.use("dark_background")

# =============================================================================
# STEP 3: Baseline 1 -- "Always Up"
# =============================================================================
always_up_predictions = np.ones(len(test), dtype=int)

always_up_accuracy = accuracy_score(y_true, always_up_predictions)
always_up_f1       = f1_score(y_true, always_up_predictions, zero_division=0)

print("\n=== Baseline 1: Always Up ===")
print(f"Accuracy: {always_up_accuracy:.4f} ({always_up_accuracy*100:.2f}%)")
print(f"F1 Score: {always_up_f1:.4f}")
print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_true, always_up_predictions))

# =============================================================================
# STEP 4: Baseline 2 -- "Persistence"
# =============================================================================
today_direction = (test["Return"] > 0).astype(int)
persistence_predictions = today_direction.values

persistence_accuracy = accuracy_score(y_true, persistence_predictions)
persistence_f1       = f1_score(y_true, persistence_predictions, zero_division=0)

print("\n=== Baseline 2: Persistence (yesterday's direction) ===")
print(f"Accuracy: {persistence_accuracy:.4f} ({persistence_accuracy*100:.2f}%)")
print(f"F1 Score: {persistence_f1:.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_true, persistence_predictions))

# =============================================================================
# STEP 5: Save results to CSV
# =============================================================================
results_dir = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)

results = pd.DataFrame([
    {
        "model":      "Naive Baseline: Always Up",
        "accuracy":   round(always_up_accuracy, 4),
        "f1_score":   round(always_up_f1, 4),
        "test_start": test.index.min().date(),
        "test_end":   test.index.max().date(),
        "notes":      "Predicts up every day. Sets the accuracy floor (~53.8%)."
    },
    {
        "model":      "Naive Baseline: Persistence",
        "accuracy":   round(persistence_accuracy, 4),
        "f1_score":   round(persistence_f1, 4),
        "test_start": test.index.min().date(),
        "test_end":   test.index.max().date(),
        "notes":      "Predicts tomorrow = today's direction. Tests simple momentum."
    }
])

results_path = os.path.join(results_dir, "model_comparison.csv")
results.to_csv(results_path, index=False)
print(f"\nSaved results to {results_path}")
print(results.to_string(index=False))

# =============================================================================
# STEP 6: Visualize
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

models_list = ["Always Up", "Persistence"]
accuracies  = [always_up_accuracy, persistence_accuracy]
colors      = ["steelblue", "darkorange"]

axes[0].bar(models_list, [a * 100 for a in accuracies], color=colors, width=0.4)
axes[0].axhline(y=50, color="red", linestyle="--", linewidth=1, label="50% random chance")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Naive Baseline Accuracy on Test Set (2021-2026)")
axes[0].set_ylim(45, 65)
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

for i, (m, acc) in enumerate(zip(models_list, accuracies)):
    axes[0].text(i, acc * 100 + 0.3, f"{acc*100:.2f}%", ha="center", fontsize=11, fontweight="bold")

axes[1].plot(test.index, test["Return"] * 100, linewidth=0.5, color="gray", alpha=0.7)
axes[1].axhline(y=0, color="black", linewidth=0.8)
axes[1].set_ylabel("Daily Return (%)")
axes[1].set_title("S&P 500 Daily Returns During Test Period (context)")
axes[1].set_xlabel("Date")
axes[1].grid(alpha=0.3)

plt.tight_layout()
figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)
fig_path = os.path.join(figures_dir, "day6_baseline_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to {fig_path}")
plt.style.use("dark_background")
print("Day 6 complete.")