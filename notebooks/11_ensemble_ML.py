# =============================================================================
# PART 11: RANDOM FOREST + XGBOOST
# =============================================================================
# Research Question:
# To what extent do increasingly complex machine learning models improve
# predictive performance compared to classical statistical AND mathematical
# methods when forecasting financial time-series data?
#
# What these models are:
# These are the COMPLEX ML models -- non-linear ensemble methods that can
# capture feature interactions and non-linear patterns that linear models miss.
#
# Random Forest:
#   Builds hundreds of decision trees, each trained on a random subset of
#   data and features. Takes majority vote across all trees for final
#   prediction. The randomness makes each tree slightly different so errors
#   cancel out when averaged -- this is called "bagging" (bootstrap aggregating).
#
# XGBoost (Extreme Gradient Boosting):
#   Builds trees SEQUENTIALLY -- each new tree is specifically trained to
#   correct the mistakes the previous trees made. This is called gradient
#   boosting. More powerful than Random Forest but more sensitive to tuning.
#
# Why these are "complex ML":
#   Unlike Logistic/Ridge which find a single linear boundary, these models
#   capture non-linear relationships and feature interactions:
#   e.g. "when volatility is high AND return_lag_1 is negative → predict down"
#   Linear models cannot represent this kind of conditional logic.
#
# Key question for the research:
#   Logistic Regression got 54.40% (+0.47pp above baseline).
#   Do non-linear ensemble methods do meaningfully better?
#   This is what Part 11 answers.
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"]  = "#0e0e0e"
plt.rcParams["axes.facecolor"]    = "#1a1a1a"
plt.rcParams["savefig.facecolor"] = "#0e0e0e"

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

script_dir    = os.path.dirname(os.path.abspath(__file__))
features_path = os.path.join(script_dir, "..", "data processed", "sp500_features.csv")

data = pd.read_csv(features_path, index_col=0, parse_dates=True)
data = data.dropna()

print(f"Data loaded: {data.index[0].date()} to {data.index[-1].date()} "
      f"({len(data)} rows)")
print(f"Columns: {list(data.columns)}\n")

# =============================================================================
# STEP 2: DEFINE FEATURES AND TARGETS
# =============================================================================
# Same feature set as Part 10 -- keeping it consistent across all ML models
# so results are directly comparable.

feature_cols = [
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "return_ma_5",  "return_ma_10",
    "volatility_5", "volatility_10", "volatility_20"
]

X           = data[feature_cols]
y_direction = data["target_direction"]

print(f"Features: {feature_cols}")
print(f"X shape: {X.shape}")
print(f"Class balance: {y_direction.mean():.4f} (fraction of up days)\n")

# =============================================================================
# STEP 3: WALK-FORWARD VALIDATION SETUP
# =============================================================================

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

# =============================================================================
# STEP 4: WALK-FORWARD LOOP
# =============================================================================
# NOTE: Tree-based models (Random Forest, XGBoost) do NOT need StandardScaler.
# Unlike linear models, trees split on feature values using thresholds --
# the absolute scale of features doesn't affect which splits are chosen.
# Scaling would not hurt but is unnecessary here.

rf_results  = []
xgb_results = []

for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):

    X_train     = X.iloc[train_start:train_end]
    X_test      = X.iloc[test_start:test_end]
    y_train     = y_direction.iloc[train_start:train_end]
    y_test      = y_direction.iloc[test_start:test_end]

    print(f"Fold {fold_idx + 1}:")
    print(f"  Train: {X.index[train_start].date()} → "
          f"{X.index[train_end - 1].date()} ({train_end - train_start} rows)")
    print(f"  Test:  {X.index[test_start].date()} → "
          f"{X.index[test_end - 1].date()} ({test_end - test_start} rows)")

    # -------------------------------------------------------------------------
    # RANDOM FOREST
    # -------------------------------------------------------------------------
    # n_estimators=200: build 200 trees (more trees = more stable, diminishing
    #                   returns beyond ~200 for this dataset size)
    # max_depth=4: limit tree depth to prevent overfitting. Shallow trees
    #              generalize better on noisy financial data.
    # min_samples_leaf=20: each leaf must have at least 20 samples -- prevents
    #                      the model from memorizing tiny patterns in training.
    # max_features="sqrt": each tree only sees sqrt(n_features) features --
    #                      this is what makes each tree different from others.
    # random_state=42: reproducibility
    # n_jobs=-1: use all CPU cores to train trees in parallel

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1  = f1_score(y_test, rf_preds, zero_division=0)

    # -------------------------------------------------------------------------
    # XGBOOST
    # -------------------------------------------------------------------------
    # n_estimators=200: number of boosting rounds (trees built sequentially)
    # max_depth=3: shallower than RF -- XGBoost boosting compensates for depth
    # learning_rate=0.05: how much each tree corrects the previous. Smaller =
    #                     more conservative, needs more trees, generally better.
    # subsample=0.8: each tree trained on 80% of rows (adds randomness)
    # colsample_bytree=0.8: each tree sees 80% of features (adds randomness)
    # use_label_encoder=False, eval_metric="logloss": suppress warnings
    # random_state=42: reproducibility

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_f1  = f1_score(y_test, xgb_preds, zero_division=0)

    print(f"  Random Forest → Accuracy: {rf_acc*100:.2f}% | F1: {rf_f1:.4f}")
    print(f"  XGBoost       → Accuracy: {xgb_acc*100:.2f}% | F1: {xgb_f1:.4f}\n")

    rf_results.append({
        "fold":       fold_idx + 1,
        "test_start": X.index[test_start].date(),
        "test_end":   X.index[test_end - 1].date(),
        "accuracy":   rf_acc,
        "f1_score":   rf_f1,
    })

    xgb_results.append({
        "fold":       fold_idx + 1,
        "test_start": X.index[test_start].date(),
        "test_end":   X.index[test_end - 1].date(),
        "accuracy":   xgb_acc,
        "f1_score":   xgb_f1,
    })

# =============================================================================
# STEP 5: AGGREGATE RESULTS
# =============================================================================

rf_df  = pd.DataFrame(rf_results)
xgb_df = pd.DataFrame(xgb_results)

mean_rf_acc  = rf_df["accuracy"].mean()
mean_rf_f1   = rf_df["f1_score"].mean()
mean_xgb_acc = xgb_df["accuracy"].mean()
mean_xgb_f1  = xgb_df["f1_score"].mean()

naive_baseline = 0.5393
logistic_acc   = 0.5440  # from Part 10

print("=" * 60)
print("PART 11 SUMMARY ACROSS ALL FOLDS")
print("=" * 60)
print(f"Random Forest:")
print(f"  Mean Accuracy:  {mean_rf_acc*100:.2f}%")
print(f"  Mean F1:        {mean_rf_f1:.4f}")
print(f"  vs Baseline:    {(mean_rf_acc - naive_baseline)*100:+.2f} pp")
print(f"  vs Logistic:    {(mean_rf_acc - logistic_acc)*100:+.2f} pp")
print()
print(f"XGBoost:")
print(f"  Mean Accuracy:  {mean_xgb_acc*100:.2f}%")
print(f"  Mean F1:        {mean_xgb_f1:.4f}")
print(f"  vs Baseline:    {(mean_xgb_acc - naive_baseline)*100:+.2f} pp")
print(f"  vs Logistic:    {(mean_xgb_acc - logistic_acc)*100:+.2f} pp")
print("=" * 60)

# =============================================================================
# STEP 6: FEATURE IMPORTANCE
# =============================================================================
# Both models give feature importance scores.
# Random Forest: average reduction in impurity across all trees
# XGBoost: average gain from splits using each feature
# Both are on different scales so we normalize to sum to 1.

# Refit on full training data for stable importance scores
train_end_full  = folds[-1][1]
X_train_full    = X.iloc[:train_end_full]
y_train_full    = y_direction.iloc[:train_end_full]

rf_full = RandomForestClassifier(
    n_estimators=200, max_depth=4, min_samples_leaf=20,
    max_features="sqrt", random_state=42, n_jobs=-1
)
rf_full.fit(X_train_full, y_train_full)

xgb_full = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss", verbosity=0
)
xgb_full.fit(X_train_full, y_train_full)

rf_importance  = rf_full.feature_importances_
xgb_importance = xgb_full.feature_importances_

importance_df = pd.DataFrame({
    "feature":        feature_cols,
    "rf_importance":  rf_importance,
    "xgb_importance": xgb_importance,
}).sort_values("rf_importance", ascending=False)

print("\nFeature Importance:")
print(importance_df[["feature", "rf_importance", "xgb_importance"]].to_string(index=False))

# =============================================================================
# STEP 7: SAVE RESULTS TO model_comparison.csv
# =============================================================================

results_dir     = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)
comparison_path = os.path.join(results_dir, "model_comparison.csv")

new_rows = pd.DataFrame([
    {
        "model":      "Random Forest",
        "accuracy":   round(mean_rf_acc, 4),
        "f1_score":   round(mean_rf_f1, 4),
        "rmse":       None,
        "mae":        None,
        "test_start": rf_results[0]["test_start"],
        "test_end":   rf_results[-1]["test_end"],
        "notes":      "n_estimators=200, max_depth=4, min_samples_leaf=20. "
                      "No scaling needed for tree models."
    },
    {
        "model":      "XGBoost",
        "accuracy":   round(mean_xgb_acc, 4),
        "f1_score":   round(mean_xgb_f1, 4),
        "rmse":       None,
        "mae":        None,
        "test_start": xgb_results[0]["test_start"],
        "test_end":   xgb_results[-1]["test_end"],
        "notes":      "n_estimators=200, max_depth=3, lr=0.05, subsample=0.8. "
                      "Gradient boosting sequential ensemble."
    },
])

if os.path.exists(comparison_path):
    existing = pd.read_csv(comparison_path)
    existing = existing[~existing["model"].isin(["Random Forest", "XGBoost"])]
    combined = pd.concat([existing, new_rows], ignore_index=True)
else:
    combined = new_rows

combined.to_csv(comparison_path, index=False)
print(f"\nSaved results to: {comparison_path}")

# =============================================================================
# STEP 8: VISUALIZATIONS
# =============================================================================

figures_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Part 11: Random Forest + XGBoost Results",
             fontsize=14, fontweight="bold", color="white")

# --- Plot 1: Accuracy per fold (both models) ---
ax1       = axes[0]
fold_nums = rf_df["fold"].tolist()
x         = np.arange(len(fold_nums))
width     = 0.35

bars_rf  = ax1.bar(x - width/2, rf_df["accuracy"] * 100,
                   width, label="Random Forest", color="#5ce0a0", alpha=0.85)
bars_xgb = ax1.bar(x + width/2, xgb_df["accuracy"] * 100,
                   width, label="XGBoost", color="#e0c45c", alpha=0.85)

ax1.axhline(naive_baseline * 100, color="white", linestyle="--",
            linewidth=1.5, label=f"Naive baseline ({naive_baseline*100:.1f}%)")
ax1.axhline(logistic_acc * 100, color="#aaaaaa", linestyle=":",
            linewidth=1.2, label=f"Logistic ({logistic_acc*100:.1f}%)")
ax1.set_xlabel("Fold", color="white")
ax1.set_ylabel("Direction Accuracy (%)", color="white")
ax1.set_title("Accuracy per Fold", color="white")
ax1.set_xticks(x)
ax1.set_xticklabels(fold_nums)
ax1.legend(fontsize=7)
ax1.set_ylim(40, 65)
ax1.tick_params(colors="white")

# --- Plot 2: Feature importance comparison ---
ax2 = axes[1]
y_pos = np.arange(len(feature_cols))
imp_sorted = importance_df.sort_values("rf_importance", ascending=True)

ax2.barh(y_pos - 0.2, imp_sorted["rf_importance"],
         height=0.35, label="Random Forest", color="#5ce0a0", alpha=0.85)
ax2.barh(y_pos + 0.2, imp_sorted["xgb_importance"],
         height=0.35, label="XGBoost", color="#e0c45c", alpha=0.85)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(imp_sorted["feature"], fontsize=8, color="white")
ax2.set_xlabel("Feature Importance", color="white")
ax2.set_title("Feature Importance\nRF vs XGBoost", color="white")
ax2.legend(fontsize=8)
ax2.tick_params(colors="white")

# --- Plot 3: All models comparison ---
ax3        = axes[2]
models     = ["Always Up\n(Naive)", "ARIMA", "GARCH", "GBM+\nMC",
              "Logistic", "Ridge", "Random\nForest", "XGBoost"]
accuracies = [53.93, 50.73, 49.23, 52.10,
              54.40, 50.77,
              mean_rf_acc * 100, mean_xgb_acc * 100]
bar_colors = ["#888888", "#e0a85c", "#5cb8e0", "#a05ce0",
              "#5ce0a0", "#e0c45c", "#60c0e0", "#e08060"]
bars3      = ax3.bar(models, accuracies, color=bar_colors, alpha=0.85)
ax3.axhline(naive_baseline * 100, color="white", linestyle="--",
            linewidth=1.5, label=f"Naive floor ({naive_baseline*100:.1f}%)")
ax3.axhline(50, color="gray", linestyle=":", linewidth=1)
ax3.set_ylabel("Direction Accuracy (%)", color="white")
ax3.set_title("All Models So Far", color="white")
ax3.set_ylim(40, 65)
ax3.legend(fontsize=7)
ax3.tick_params(colors="white", labelsize=7)
for bar, acc in zip(bars3, accuracies):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{acc:.1f}%", ha="center", va="bottom", fontsize=7, color="white")

plt.tight_layout()
fig_path = os.path.join(figures_dir, "part11_ensemble_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to: {fig_path}")

# =============================================================================
# STEP 9: FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Random Forest:")
print(f"  Mean accuracy:     {mean_rf_acc*100:.2f}%")
print(f"  Mean F1:           {mean_rf_f1:.4f}")
print(f"  vs Naive baseline: {(mean_rf_acc - naive_baseline)*100:+.2f} pp")
print(f"  vs Logistic Reg:   {(mean_rf_acc - logistic_acc)*100:+.2f} pp")
print()
print(f"XGBoost:")
print(f"  Mean accuracy:     {mean_xgb_acc*100:.2f}%")
print(f"  Mean F1:           {mean_xgb_f1:.4f}")
print(f"  vs Naive baseline: {(mean_xgb_acc - naive_baseline)*100:+.2f} pp")
print(f"  vs Logistic Reg:   {(mean_xgb_acc - logistic_acc)*100:+.2f} pp")
print("=" * 60)
