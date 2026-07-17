# What these models are:
# These are the SIMPLEST real ML models -- they form the entry point into
# the machine learning tradition in this project.
#
# Logistic Regression (classification):
#   Predicts direction (1=up, 0=down). Finds a LINEAR boundary separating
#   up days from down days in feature space. However, it is a
#   classifier not a regressor.
#
# Ridge Regression (regression):
#   Predicts the actual next-day return value (continuous output).
#   Standard linear regression + L2 regularization penalty to prevent
#   overfitting by keeping weights small.
#
# The core equation for both:
#   prediction = w1*x1 + w2*x2 + ... + wn*xn + b
#   Where x = features, w = learned weights, b = bias term.
#   This is a weighted sum -- completely linear.
#
# Regularization:
#   Without it, models can overfit by assigning huge weights to features.
#   Ridge adds a penalty = alpha * sum(w²) that keeps all weights small.
#   Logistic uses C (inverse of regularization strength): smaller C = stronger
#   regularization. We tune both using cross-validation.
#
# Why these models matter for the research question:
#   If logistic regression can't beat 53.93% (naive baseline), it means
#   linear relationships in these features have no predictive power.
#   Random Forest, XGBoost, and LSTM then get to show if non-linearity helps.
#   The progression from linear → non-linear → deep learning is the core
#   narrative of this project.


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"]  = "#0e0e0e"
plt.rcParams["axes.facecolor"]    = "#1a1a1a"
plt.rcParams["savefig.facecolor"] = "#0e0e0e"


# STEP 1: LOAD DATA


script_dir    = os.path.dirname(os.path.abspath(__file__))
features_path = os.path.join(script_dir, "..", "data processed", "sp500_features.csv")

data = pd.read_csv(features_path, index_col=0, parse_dates=True)
data = data.dropna()

print(f"Data loaded: {data.index[0].date()} to {data.index[-1].date()} "
      f"({len(data)} rows)")
print(f"Columns: {list(data.columns)}\n")


# STEP 2: DEFINE FEATURES AND TARGETS



feature_cols = [
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "return_ma_5",  "return_ma_10",
    "volatility_5", "volatility_10", "volatility_20"
]

X = data[feature_cols]
y_direction = data["target_direction"]   # classification target (1=up, 0=down)
y_return    = data["target_return"]      # regression target (continuous return)

print(f"Features used: {feature_cols}")
print(f"X shape: {X.shape}")
print(f"Class balance: {y_direction.mean():.4f} (fraction of up days)\n")


# STEP 3: WALK-FORWARD VALIDATION SETUP

# Same 5-fold expanding window setup used in Parts 8 and 9.

n         = len(X)
n_splits  = 5
test_size = int(n * 0.10)
min_train = int(n * 0.50)

folds = []
for i in range(n_splits):
    test_end   = n - (n_splits - 1 - i) * test_size
    test_start = test_end - test_size
    train_end  = test_start
    if train_end < min_train:
        continue
    folds.append((0, train_end, test_start, test_end))

print(f"Walk-forward validation: {len(folds)} folds\n")


# STEP 4: WALK-FORWARD LOOP

# For each fold:
#   1. Split into train/test
#   2. Scale features (CRITICAL for linear models -- explained below)
#   3. Fit Logistic Regression on train, predict direction on test
#   4. Fit Ridge Regression on train, predict return on test
#   5. Evaluate both models

# WHY SCALING IS CRITICAL FOR LINEAR MODELS:
# Features have very different scales:
#   - return_lag_1 might be 0.001 (tiny decimal)
#   - volatility_20 might be 0.015 (slightly bigger)
# Without scaling, the model gives more weight to larger-valued features
# just because of their scale, not their actual predictive power.
# StandardScaler transforms each feature to mean=0, std=1.
# IMPORTANT: fit the scaler ONLY on training data, then apply to test.
# Fitting on test data would leak future information (lookahead bias).

logistic_results = []
ridge_results    = []

for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):

    X_train      = X.iloc[train_start:train_end]
    X_test       = X.iloc[test_start:test_end]
    y_dir_train  = y_direction.iloc[train_start:train_end]
    y_dir_test   = y_direction.iloc[test_start:test_end]
    y_ret_train  = y_return.iloc[train_start:train_end]
    y_ret_test   = y_return.iloc[test_start:test_end]

    print(f"Fold {fold_idx + 1}:")
    print(f"  Train: {X.index[train_start].date()} → "
          f"{X.index[train_end - 1].date()} ({train_end - train_start} rows)")
    print(f"  Test:  {X.index[test_start].date()} → "
          f"{X.index[test_end - 1].date()} ({test_end - test_start} rows)")

    # --- Scale features ---
    # Fit scaler on training data only
    scaler   = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)   # apply same scale to test


    # LOGISTIC REGRESSION (direction classification)

    # C=0.1: moderate regularization (smaller C = stronger penalty)
    # max_iter=1000: allow enough iterations to converge
    # random_state=42: reproducibility
    log_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    log_model.fit(X_train_scaled, y_dir_train)
    log_preds = log_model.predict(X_test_scaled)

    log_acc = accuracy_score(y_dir_test, log_preds)
    log_f1  = f1_score(y_dir_test, log_preds, zero_division=0)

    # RIDGE REGRESSION (return prediction)

    # alpha=1.0: standard regularization strength starting point
    # We use Ridge to predict continuous return values
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_ret_train)
    ridge_preds = ridge_model.predict(X_test_scaled)

    ridge_rmse = np.sqrt(mean_squared_error(y_ret_test, ridge_preds))
    ridge_mae  = mean_absolute_error(y_ret_test, ridge_preds)

    # Also convert Ridge predictions to direction for comparison
    ridge_dir_preds = (ridge_preds > 0).astype(int)
    ridge_dir_acc   = accuracy_score(y_dir_test, ridge_dir_preds)
    ridge_dir_f1    = f1_score(y_dir_test, ridge_dir_preds, zero_division=0)

    print(f"  Logistic Regression → Accuracy: {log_acc*100:.2f}% | F1: {log_f1:.4f}")
    print(f"  Ridge Regression    → Dir Acc:  {ridge_dir_acc*100:.2f}% | "
          f"RMSE: {ridge_rmse:.6f} | MAE: {ridge_mae:.6f}\n")

    logistic_results.append({
        "fold":         fold_idx + 1,
        "test_start":   X.index[test_start].date(),
        "test_end":     X.index[test_end - 1].date(),
        "accuracy":     log_acc,
        "f1_score":     log_f1,
    })

    ridge_results.append({
        "fold":         fold_idx + 1,
        "test_start":   X.index[test_start].date(),
        "test_end":     X.index[test_end - 1].date(),
        "dir_accuracy": ridge_dir_acc,
        "f1_score":     ridge_dir_f1,
        "rmse":         ridge_rmse,
        "mae":          ridge_mae,
    })


# STEP 5: AGGREGATE RESULTS


log_df   = pd.DataFrame(logistic_results)
ridge_df = pd.DataFrame(ridge_results)

mean_log_acc      = log_df["accuracy"].mean()
mean_log_f1       = log_df["f1_score"].mean()
mean_ridge_acc    = ridge_df["dir_accuracy"].mean()
mean_ridge_f1     = ridge_df["f1_score"].mean()
mean_ridge_rmse   = ridge_df["rmse"].mean()
mean_ridge_mae    = ridge_df["mae"].mean()

naive_baseline = 0.5393

print("=" * 60)
print("PART 10 SUMMARY ACROSS ALL FOLDS")
print("=" * 60)
print(f"Logistic Regression:")
print(f"  Mean Accuracy:  {mean_log_acc*100:.2f}%")
print(f"  Mean F1:        {mean_log_f1:.4f}")
print(f"  vs Baseline:    {(mean_log_acc - naive_baseline)*100:+.2f} pp")
print()
print(f"Ridge Regression (direction from sign of predicted return):")
print(f"  Mean Dir Acc:   {mean_ridge_acc*100:.2f}%")
print(f"  Mean F1:        {mean_ridge_f1:.4f}")
print(f"  Mean RMSE:      {mean_ridge_rmse:.6f}")
print(f"  Mean MAE:       {mean_ridge_mae:.6f}")
print(f"  vs Baseline:    {(mean_ridge_acc - naive_baseline)*100:+.2f} pp")
print("=" * 60)


# STEP 6: FEATURE IMPORTANCE


train_end_full = folds[-1][1]
X_train_full   = X.iloc[:train_end_full]
y_dir_full     = y_direction.iloc[:train_end_full]

scaler_full    = StandardScaler()
X_train_scaled_full = scaler_full.fit_transform(X_train_full)

log_full = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
log_full.fit(X_train_scaled_full, y_dir_full)

coef_df = pd.DataFrame({
    "feature":     feature_cols,
    "coefficient": log_full.coef_[0],
    "abs_coef":    np.abs(log_full.coef_[0])
}).sort_values("abs_coef", ascending=False)

print("\nLogistic Regression Feature Importance (by absolute coefficient):")
print(coef_df[["feature", "coefficient"]].to_string(index=False))


# STEP 7: SAVE RESULTS TO model_comparison.csv


results_dir     = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)
comparison_path = os.path.join(results_dir, "model_comparison.csv")

new_rows = pd.DataFrame([
    {
        "model":      "Logistic Regression",
        "accuracy":   round(mean_log_acc, 4),
        "f1_score":   round(mean_log_f1, 4),
        "rmse":       None,
        "mae":        None,
        "test_start": logistic_results[0]["test_start"],
        "test_end":   logistic_results[-1]["test_end"],
        "notes":      f"C=0.1, StandardScaler, features: {feature_cols}. "
                      f"Simple ML classification baseline."
    },
    {
        "model":      "Ridge Regression",
        "accuracy":   round(mean_ridge_acc, 4),
        "f1_score":   round(mean_ridge_f1, 4),
        "rmse":       round(mean_ridge_rmse, 6),
        "mae":        round(mean_ridge_mae, 6),
        "test_start": ridge_results[0]["test_start"],
        "test_end":   ridge_results[-1]["test_end"],
        "notes":      f"alpha=1.0, StandardScaler, direction from sign of predicted return. "
                      f"Simple ML regression baseline."
    },
])

if os.path.exists(comparison_path):
    existing = pd.read_csv(comparison_path)
    existing = existing[~existing["model"].isin(["Logistic Regression", "Ridge Regression"])]
    combined = pd.concat([existing, new_rows], ignore_index=True)
else:
    combined = new_rows

combined.to_csv(comparison_path, index=False)
print(f"\nSaved results to: {comparison_path}")


# STEP 8: VISUALIZATIONS

figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Part 10: Logistic + Ridge Regression Results",
             fontsize=14, fontweight="bold", color="white")

# --- Plot 1: Logistic Regression accuracy per fold ---
ax1       = axes[0]
fold_nums = log_df["fold"].tolist()
log_accs  = log_df["accuracy"].tolist()
colors    = ["#e05c5c" if a < naive_baseline else "#5ce0a0" for a in log_accs]
bars      = ax1.bar(fold_nums, [a * 100 for a in log_accs], color=colors, alpha=0.85)
ax1.axhline(naive_baseline * 100, color="white", linestyle="--",
            linewidth=1.5, label=f"Naive baseline ({naive_baseline*100:.1f}%)")
ax1.axhline(50, color="gray", linestyle=":", linewidth=1, label="50% random")
ax1.set_xlabel("Fold", color="white")
ax1.set_ylabel("Direction Accuracy (%)", color="white")
ax1.set_title("Logistic Regression\nAccuracy per Fold", color="white")
ax1.set_xticks(fold_nums)
ax1.legend(fontsize=8)
ax1.set_ylim(40, 65)
ax1.tick_params(colors="white")
for bar, acc in zip(bars, log_accs):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=8, color="white")

# --- Plot 2: Feature importance ---
ax2 = axes[1]
colors_coef = ["#5ce0a0" if c > 0 else "#e05c5c" for c in coef_df["coefficient"]]
ax2.barh(coef_df["feature"], coef_df["coefficient"], color=colors_coef, alpha=0.85)
ax2.axvline(0, color="white", linewidth=0.8)
ax2.set_xlabel("Coefficient Value", color="white")
ax2.set_title("Logistic Regression\nFeature Coefficients", color="white")
ax2.tick_params(colors="white")
ax2.invert_yaxis()

# --- Plot 3: All models comparison ---
ax3        = axes[2]
models     = ["Always Up\n(Naive)", "ARIMA\n(1,0,1)", "GARCH\n(1,1)",
              "GBM+\nMC", "Logistic\nReg", "Ridge\nReg"]
accuracies = [53.93, 50.73, 49.23, 52.10,
              mean_log_acc * 100, mean_ridge_acc * 100]
bar_colors = ["#888888", "#e0a85c", "#5cb8e0", "#a05ce0", "#5ce0a0", "#e0c45c"]
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
             f"{acc:.1f}%", ha="center", va="bottom", fontsize=7, color="white")

plt.tight_layout()
fig_path = os.path.join(figures_dir, "part10_linear_ml_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to: {fig_path}")


# STEP 9: FINAL SUMMARY


print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Logistic Regression:")
print(f"  Mean accuracy:       {mean_log_acc*100:.2f}%")
print(f"  Mean F1:             {mean_log_f1:.4f}")
print(f"  vs Naive baseline:   {(mean_log_acc - naive_baseline)*100:+.2f} pp")
print()
print(f"Ridge Regression:")
print(f"  Mean dir accuracy:   {mean_ridge_acc*100:.2f}%")
print(f"  Mean F1:             {mean_ridge_f1:.4f}")
print(f"  Mean RMSE:           {mean_ridge_rmse:.6f}")
print(f"  Mean MAE:            {mean_ridge_mae:.6f}")
print(f"  vs Naive baseline:   {(mean_ridge_acc - naive_baseline)*100:+.2f} pp")
print("=" * 60)
