## Week 1 Plan (Data + Foundations)

**Target/asset decision (locked in):** S&P 500 index (^GSPC), daily data,
2000-present. Primary target: next-day direction (up/down). Secondary
target: next-day return (for regression models).

**Why these choices:** Index data avoids single-company noise and is the
standard benchmark in the forecasting literature. Direction prediction is
more forgiving to evaluate for a first project; return prediction adds
depth, so we keep both.

### (2026-06-14) Part 1: Data Loading and Exploration
- What I did: Downloaded S&P 500 daily price data from 2000 to present (6641 rows). i did this by importing yfinance into my Visual Studio python code. During calling the data I found a bug where the yfinance returned data in MultiIndex columns so I had to flatten it using pandas. I used the pandas' pct_change() function to compute daily returns from the price data. Used matplotlib to generate and review plots of price, returns, and a histogram of returns. Saved raw and processed data as CSVs using pandas.
- What I learned: Raw price trends upward over decades with no fixed "level" (non-stationary), while daily returns hover around zero the whole time (stationary). This is why financial models work with returns instead of raw price. Also saw that the histograp of returns has fatter tails than a normal distribution would predict (confirmed numerically using pandas" built-in skew() and kurt() functions). 
- Problems: Had a path error where the script couldn't find the saved CSV depending on which folder the terminal was running from -- fixed using Python's OS library to make the script locate files relative to its own location instead of the current working directory.


### (2026-06-15) Part 2: Stationarity and Autocorrelation
- What I did: Used the statsmodels library's adfuller() function to run the Augmented Dickey-Fuller (ADF) test on price and on returns. Used statsmodels' plot_acf() and plot_pacf() funcitons to generate autocorrelation plots for returns, amd separately for squared returns (computed using basic Python exponentiation, Return**2).
- What I learned: The ADF test on price gave a p-value of 1.0000 (no evidence of stationarity). The ADF test on returns gave a p-value of about 0.000000 (strong evidence of stationarity). The ACF/PACF plots on raw returns showed almost no statistically meaningful autocorrelation at any lag, meaning past returns don't reliably  predict the direction of future returns. The ACF/PACF plots on squared returns showed strong, statistically meaninful autocorrelation at every lag tested (around 0.40 at lag 2), meaning past volatility. This is called volatility clustering.
- Problems: None once the path issue from Day 1 was fixed (same fix carried over using the os library).

### (2026-06-16) Part 3: Feature Engineering
- What I did: Used pandas' shift() and rolling() functions to build engineered features -- lagged returns (1,2,3, and 5 days back), rolling averages (5 day and 10-day), and rolling volatility (5 day, 10-day, and 20-day standard deviation of returns). Defined two prediction targets using shift(-1) target_direction (1 if tomorrow's return is positive, 0 otherwise) and target_return (tomorrow's actual return value). Dropped rows with missing values created by the lag/rolling windows. Saved the result as sp500_features.csv.
- What I learned: Why features must only use past data (lookahead bias/data leakage) --shift(1) pulls past data (safe for features), while shift(-1) pulls future data (only safe for defining the target, since that's literally the answer we're trying to predict, not an input). Verified this manually: return_lag_1 on any row exactly matches Return from the row before it, and target_return on any row exactly matches Return from the row after it -- confirming the shifts work correctly in both directions.
- Results: Final dataset has 6619 rows and 17 columns after dropping 21 rows with missing values. Class balance for target_direction: 53.8% up-days vs 46.2% down-days -- mildly imbalanced, meaning a model that always predicts "up" would get ~53.8% accuracy without learning anything real. This becomes the baseline every real model needs to beat by a meaningful, statistically significant margin.
- Problems: None -- ran cleanly once the file was saved into the correct folder.

### (2026-06-17) Part 4: Walk-Forward Train/Test Split (complete)
- What I did: First tested the WRONG way to split time-series data --  randomly shuffling rows before splitting into train/test (the default behavior of sklearn's train_test_split). Then did it the RIGHT way: a single chronological split where all training data comes before all test data, with no overlap. Then built a more rigorous version called walk-forward validation using sklearn's TimeSeriesSplit, which repeats the chronological split 5 times with an expanding training window, giving 5 separate train/test pairs (folds) instead of just one.
  
- What I learned: When data is randomly shuffled, both the resulting "training" pile and "testing" pile end up spanning the entire 26-year date range -- confirmed directly in my own output (both showed date ranges from 2000 to 2026). This means a model could secretly train on recent data (e.g. 2024) and get "tested" on older data (e.g. 2010), which is impossible in real life and makes the evaluation meaningless. The chronological split fixes this: training data always comes before test data in time, so the model is never trained on the future and tested on the past -- only ever the other way around. Walk-forward validation repeats this idea 5 times, each fold's test period starting exactly one day after that fold's training period ends, giving a more trustworthy evaluation than relying on just one split.
  
- Results: Chronological 80/20 split: train = 2000-02-02 to 2021-02-18 (5295 rows), test = 2021-02-19 to 2026-05-28 (1324 rows), no overlap. 5-fold walk-forward validation confirmed correct: e.g. Fold 3 train ends 2013-04-02, Fold 3 test starts 2013-04-03 -- exactly adjacent, no gap, no overlap.
  
- Problems: Initially found the terminology (row, fold, train/test set) confusing when looking at the output all at once. Worked through it by focusing on one number at a time instead of the whole output -- e.g. comparing just the WRONG approach's two date ranges side by side first, before moving to the RIGHT approach.

### (2026-06-18) Part 5: Literature Review
- What I did: Read and logged 12+ papers across 7 themes into references/literature_review_notes.md. Themes covered:
  - ML vs. classical methods broad comparisons (Makridakis 2018, Springer Nature 2025, MDPI Entropy 2025)
  - ARIMA vs. LSTM on S&P 500 specifically (Pilla & Mekonen 2025, SCITEPRESS 2023, JRFM 2026)
  - Random Forest and XGBoost vs. classical (BCP 2023, BCP 2022)
  - Efficient Market Hypothesis theoretical backdrop (Fama 1970, Southampton University study)
  - Fischer & Krauss 2018 -- the most cited LSTM finance paper
  - Foundational GARCH papers (Engle 1982, Bollerslev 1986, Marisetty 2024)
  - Geometric Brownian Motion (Samuelson 1965, Black & Scholes 1973)
  
- What I learned:
  1. ML does not consistently beat classical methods -- results depend heavily on methodology and evaluation setup.
  2. Predicting price LEVEL is much easier than predicting return DIRECTION -- many "impressive" ML papers are doing the easier task.
  3. GARCH's niche is volatility forecasting, not direction prediction -- consistent with Day 2 squared-returns ACF finding.
  4. EMH (Fama 1970) and GBM (Samuelson 1965) are the theoretical reason one-day-ahead return forecasting is hard for any model.
  5. Fischer & Krauss (2018) found Random Forest outperformed LSTM in trading returns -- "more complex" does not reliably mean "better," which is exactly what this project tests.
  6. My project's genuine novelty claim: three-tradition comparison (mathematical via GBM/Monte Carlo, statistical via ARIMA/GARCH, ML via Ridge through LSTM) -- this framing does not exist cleanly in prior literature.
	

- Problems: None.

### (2026-06-19   -  2026-06-20) Part 6: Naive Baseline Models
- What I did: Built two naive baseline models evaluated on the test set 
(2021-02-19 to 2026-05-28, 1324 rows).

- Results:
  Always Up accuracy:   53.93% (F1: 0.7007) — the accuracy floor
  Persistence accuracy: 49.77% (F1: 0.5340) — below random chance

- What I learned: The persistence baseline falling below 50% confirms near-zero directional autocorrelation in returns, consistent with Day 2 ACF/PACF findings and the Efficient Market Hypothesis. There is no simple momentum to exploit. Every subsequent model must beat 53.93% to be considered meaningful.

- Saved: results/model_comparison.csv, figures/day6_baseline_results.png


### (2026-06-21) Part 7: ARIMA(1,0,1) Model
What I did: Fitted ARIMA(1,0,1) on S&P 500 daily returns using 5-fold walk-forward validation. Evaluated on RMSE, MAE, direction accuracy, and F1.

Results (mean across 5 folds):
  RMSE:               0.011698
  MAE:                0.007649
  Direction Accuracy: 50.73%
  F1 Score:           0.4236
  vs Naive Baseline:  -3.20 percentage points

What I learned: ARIMA performed worse than the naive baseline on average, confirming that past returns contain almost no linear predictive information about future direction -- consistent with Day 2 ACF/PACF and the EMH. Performance was highly unstable across folds: Fold 1-2 were below 45%, while Fold 4 reached 57%. This regime-dependence is itself a finding worth discussing in the paper.

Problems: None.

### (2026-06-22  -  2026-06-23) Part 8: GARCH

What I Did
Implemented GARCH(1,1) using the arch library with walk-forward validation across 5 folds. GARCH is a volatility model — it predicts how uncertain tomorrow will be, not which direction the market moves. To make it comparable to other models, I derived a direction signal from its volatility forecast: if predicted volatility is above the training median, predict down; if below, predict up.

Results:
Fold 1 | Test: 2013-04-09 → 2015-11-18 | Dir Accuracy: 48.26% | Vol RMSE: 0.3026% | Vol MAE: 0.2457% | F1: 0.5938
Fold 2 | Test: 2015-11-19 → 2018-07-06 | Dir Accuracy: 52.19% | Vol RMSE: 0.3015% | Vol MAE: 0.2462% | F1: 0.6238
Fold 3 | Test: 2018-07-09 → 2021-02-22 | Dir Accuracy: 47.66% | Vol RMSE: 0.4999% | Vol MAE: 0.3372% | F1: 0.4594
Fold 4 | Test: 2021-02-23 → 2023-10-06 | Dir Accuracy: 51.74% | Vol RMSE: 0.3190% | Vol MAE: 0.2463% | F1: 0.3708
Fold 5 | Test: 2023-10-09 → 2026-05-28 | Dir Accuracy: 46.29% | Vol RMSE: 0.3777% | Vol MAE: 0.2579% | F1: 0.4709

Mean Direction Accuracy : 49.23%
Mean F1 Score           : 0.5038
Mean Vol RMSE           : 0.3601%
Mean Vol MAE            : 0.2667%
vs Naive Baseline       : -4.70 pp

What I Learned
GARCH stands for Generalized AutoRegressive Conditional Heteroskedasticity.

Heteroskedasticity means variance is not constant over time — some periods are calm, some are chaotic. I confirmed this in Day 2 when I saw the squared returns ACF had significant spikes (volatility clustering).
The equation is: σ²(t) = ω + α·ε²(t−1) + β·σ²(t−1)
ω is the long-run baseline variance. α controls how much recent shocks matter. β controls how persistent the current volatility level is. In most fitted GARCH(1,1) models on financial data, α + β is close to 1.0 — meaning volatility is highly persistent.
GARCH(1,1) means one lag of the shock term and one lag of the variance term. Bollerslev introduced this in 1986 as a generalization of Engle's ARCH model (1982).

Why the direction accuracy is low and why that's fine:

GARCH was never designed to predict direction. The 49.23% direction accuracy is the result of a heuristic I imposed on top of the model — it is not what GARCH was built for. The real metric for GARCH is volatility RMSE: 0.36%. That number will be discussed in the paper as the GARCH contribution, not the direction accuracy.
The COVID fold (Fold 3, 2018–2021):

RMSE jumped from ~0.30% in normal folds to 0.50%. This is a known weakness of GARCH — it adapts slowly to sudden regime changes. The COVID crash in March 2020 was a massive, sudden volatility spike that the model had not seen in training. This is worth a sentence in the Discussion section.
How this fits the research question:

GARCH represents the classical statistical tradition alongside ARIMA. Together ARIMA and GARCH establish what the statistical tradition can do before ML models enter. So far both classical statistical models fall below the naive baseline for direction prediction — this is consistent with the Efficient Market Hypothesis and with the literature (Makridakis 2018, Fama 1970).

## (2026-06-24) Part 9 — GBM + Monte Carlo Simulation

**What I did:**
Implemented Geometric Brownian Motion + Monte Carlo simulation using the
mathematical finance tradition. For each test day, calibrated μ (drift)
and σ (volatility) from the training window, then simulated 1000 possible
next-day prices using the GBM equation. Used the median simulated price
as the direction forecast. Re-calibrated μ and σ at each step on an
expanding window to prevent lookahead bias.

---
**Results:**

Fold 1 | Test: 2013-04-09 → 2015-11-18 | Dir Accuracy: 50.38% | RMSE: 0.008133 | MAE: 0.005996 | F1: 0.5432
Fold 2 | Test: 2015-11-19 → 2018-07-06 | Dir Accuracy: 52.65% | RMSE: 0.007619 | MAE: 0.005164 | F1: 0.5741
Fold 3 | Test: 2018-07-09 → 2021-02-22 | Dir Accuracy: 54.61% | RMSE: 0.015179 | MAE: 0.009038 | F1: 0.6203
Fold 4 | Test: 2021-02-23 → 2023-10-06 | Dir Accuracy: 45.69% | RMSE: 0.011468 | MAE: 0.008572 | F1: 0.5332
Fold 5 | Test: 2023-10-09 → 2026-05-28 | Dir Accuracy: 57.19% | RMSE: 0.009652 | MAE: 0.006521 | F1: 0.6553

Mean Direction Accuracy : 52.10%
Mean F1 Score           : 0.5852
Mean Return RMSE        : 0.010410
Mean Return MAE         : 0.007058
vs Naive Baseline       : -1.83 pp

---

**What This Means:**
GBM is the best performing classical model for direction prediction --
still below the naive baseline but closer to it than ARIMA or GARCH.
Pure mathematical theory with no pattern learning came closer to the
floor than both statistical models that actually fit to the data.

Performance was extremely regime-dependent:
- Fold 3 (2018-2021): 54.61% -- beat the naive baseline, includes COVID
- Fold 4 (2021-2023): 45.69% -- worst fold, choppy post-COVID market
- Fold 5 (2023-2026): 57.19% -- best fold, strong bull market period
This is because GBM is highly sensitive to calibrated drift μ. When μ
is estimated from a long bull market history, GBM predicts "up" more
often -- which works in bull periods and fails in choppy ones.

---

**End of Phase 2-- Full Classical Scorecard:**

Model              | Accuracy | vs Baseline
Always Up (Naive)  | 53.93%   | THE FLOOR
GBM + Monte Carlo  | 52.10%   | -1.83 pp
ARIMA(1,0,1)       | 50.73%   | -3.20 pp
Persistence        | 49.77%   | -4.16 pp
GARCH(1,1)         | 49.23%   | -4.70 pp

Summary: No classical model beat the naive baseline on average.
The ML models in Phase 3 now need to beat 53.93% to justify complexity.

---

**Connections to Literature:**
- Samuelson (1965) -- GBM as the mathematical model of stock prices
- Black & Scholes (1973) -- options pricing built on top of GBM
- Fama (1970) -- EMH: results consistent with weak-form efficiency
- Makridakis (2018) -- classical methods matching ML, confirmed here


## (2026-06-26) Part 10 - Logistic Regression + Ridge Regression

 **What I Did**
Implemented the first two ML models using all 9 engineered features from
Part 3. This is the first time the full feature set was used -- all
classical models (ARIMA, GARCH, GBM) were univariate and only used the
Return column. Used 5-fold walk-forward validation with StandardScaler
applied to training data only to prevent lookahead bias.

Two models:
- Logistic Regression (C=0.1): classification -- predicts direction (up/down)
- Ridge Regression (alpha=1.0): regression -- predicts continuous return value,
  direction derived from sign of predicted return


**Results**

Fold 1 | Test: 2013-04-09 → 2015-11-18 | Logistic Acc: 54.01% | F1: 0.7002 | Ridge Dir Acc: 52.34% | RMSE: 0.008092 | MAE: 0.005969
Fold 2 | Test: 2015-11-19 → 2018-07-06 | Logistic Acc: 54.92% | F1: 0.7067 | Ridge Dir Acc: 50.08% | RMSE: 0.007580 | MAE: 0.005156
Fold 3 | Test: 2018-07-09 → 2021-02-22 | Logistic Acc: 54.92% | F1: 0.6972 | Ridge Dir Acc: 50.08% | RMSE: 0.015458 | MAE: 0.009156
Fold 4 | Test: 2021-02-23 → 2023-10-06 | Logistic Acc: 50.53% | F1: 0.6680 | Ridge Dir Acc: 49.92% | RMSE: 0.011449 | MAE: 0.008542
Fold 5 | Test: 2023-10-09 → 2026-05-28 | Logistic Acc: 57.64% | F1: 0.7297 | Ridge Dir Acc: 51.44% | RMSE: 0.009691 | MAE: 0.006566

Logistic Regression:
Mean Accuracy        : 54.40%
Mean F1 Score        : 0.7004
vs Naive Baseline    : +0.47 pp ✅ FIRST MODEL TO BEAT THE FLOOR

Ridge Regression:
Mean Dir Accuracy    : 50.77%
Mean F1 Score        : 0.5754
Mean RMSE            : 0.010454
Mean MAE             : 0.007078
vs Naive Baseline    : -3.16 pp

**Feature Importance (Logistic Regression coefficients)**

return_ma_5    : -0.044842  (strongest negative -- recent momentum predicts down)
return_lag_2   : +0.036750  (positive -- 2-day lag predicts up)
return_lag_1   : +0.029408  (positive -- yesterday predicts up)
volatility_5   : -0.027890  (negative -- high vol predicts down)
return_ma_10   : -0.027082  (negative)
volatility_10  : -0.026790  (negative)
return_lag_3   : +0.009581
return_lag_5   : +0.009090
volatility_20  : -0.001782  (weakest -- long-run vol barely matters)


**What This Means**
Logistic Regression is the FIRST model in the entire project to beat the
naive baseline -- 54.40% vs 53.93% floor (+0.47pp). Small margin but real.

Ridge Regression underperformed at 50.77%. This is expected -- Ridge is
optimized to minimize return magnitude error (RMSE), not direction accuracy.
Converting predicted returns to direction by taking the sign is a weak
heuristic. Ridge's real contribution is the RMSE metric (0.010454).

Feature story: the model found slight anti-momentum signal in return_ma_5
(negative coefficient -- recent upward trend predicts reversal) and slight
momentum in individual lags (positive coefficients on lag_1 and lag_2).
All volatility features had negative coefficients -- high volatility predicts
down days. Consistent with Day 2 findings and EMH.

Regime dependence still present:
- Fold 4 (2021-2023): 50.53% -- choppy post-COVID market killed signal
- Fold 5 (2023-2026): 57.64% -- best fold, strong bull market

 **Connections to Literature**
- Fama (1970) -- EMH: even the first ML model barely beats random
- Makridakis (2018) -- classical methods competitive with simple ML, confirmed
- Fischer & Krauss (2018) -- benchmark: LSTM got 56%, Logistic at 54.4%
  is already in the same ballpark with a far simpler model

### (2026-06-27 - 2026-06-28) Part 11 - Random Forest + XGBoost

**What I Did**
Implemented Random Forest and XGBoost using the same 9 engineered features
and 5-fold walk-forward validation as Part 10. No StandardScaler needed --
tree-based models split on feature value thresholds so scale doesn't matter.
Both models classify direction directly (up/down).

Random Forest: 200 trees, max_depth=4, min_samples_leaf=20
XGBoost: 200 rounds, max_depth=3, learning_rate=0.05, subsample=0.8

---

**Results**

Random Forest:
Fold 1 | Test: 2013-04-09 → 2015-11-18 | Accuracy: 52.65% | F1: 0.6783
Fold 2 | Test: 2015-11-19 → 2018-07-06 | Accuracy: 54.46% | F1: 0.6963
Fold 3 | Test: 2018-07-09 → 2021-02-22 | Accuracy: 54.16% | F1: 0.6873
Fold 4 | Test: 2021-02-23 → 2023-10-06 | Accuracy: 51.29% | F1: 0.6694
Fold 5 | Test: 2023-10-09 → 2026-05-28 | Accuracy: 55.37% | F1: 0.7082

Mean Accuracy     : 53.59%
Mean F1           : 0.6879
vs Naive Baseline : -0.34 pp
vs Logistic Reg   : -0.81 pp

XGBoost:
Fold 1 | Test: 2013-04-09 → 2015-11-18 | Accuracy: 51.89% | F1: 0.6232
Fold 2 | Test: 2015-11-19 → 2018-07-06 | Accuracy: 52.19% | F1: 0.6099
Fold 3 | Test: 2018-07-09 → 2021-02-22 | Accuracy: 51.44% | F1: 0.6254
Fold 4 | Test: 2021-02-23 → 2023-10-06 | Accuracy: 50.83% | F1: 0.6126
Fold 5 | Test: 2023-10-09 → 2026-05-28 | Accuracy: 53.25% | F1: 0.6570

Mean Accuracy     : 51.92%
Mean F1           : 0.6256
vs Naive Baseline : -2.01 pp
vs Logistic Reg   : -2.48 pp

---

**Feature Importance**

feature       | RF importance | XGB importance
return_ma_10  | 0.1454        | 0.1112
return_lag_1  | 0.1216        | 0.1114
return_ma_5   | 0.1182        | 0.1046
volatility_10 | 0.1140        | 0.1135
return_lag_5  | 0.1135        | 0.1050
return_lag_2  | 0.0984        | 0.1081
volatility_5  | 0.0978        | 0.1142
return_lag_3  | 0.0957        | 0.1123
volatility_20 | 0.0954        | 0.1198

---

**What This Means**
Both complex ML models underperformed Logistic Regression -- the simpler
linear model. This is one of the most important findings in the project.

Three reasons why:
1. Features are weak. All 9 features come from past returns and volatility.
   Near-zero autocorrelation means barely any signal exists regardless of
   model complexity. There is not much to find.
2. Complex models overfit more. Logistic Regression is constrained to a
   linear boundary which works better here because the true signal (if any)
   is tiny and approximately linear. RF and XGBoost have more capacity to
   memorize noise in training data.
3. Consistent with Makridakis (2018) -- complexity does not automatically
   help on noisy financial data.

Feature importance was spread nearly uniformly across all 9 features in
both models -- no single predictor dominates. This is exactly what EMH
predicts: no feature should have strong consistent predictive power.

---

**Full Scorecard After Phase 3**

Model              | Accuracy | vs Baseline
Always Up (Naive)  | 53.93%   | THE FLOOR
Logistic Regression| 54.40%   | +0.47 pp 
GBM + Monte Carlo  | 52.10%   | -1.83 pp
Random Forest      | 53.59%   | -0.34 pp
ARIMA(1,0,1)       | 50.73%   | -3.20 pp
Ridge Regression   | 50.77%   | -3.16 pp
XGBoost            | 51.92%   | -2.01 pp
Persistence        | 49.77%   | -4.16 pp
GARCH(1,1)         | 49.23%   | -4.70 pp

Only Logistic Regression beat the naive baseline so far.
LSTM now needs to answer: does deep learning change this picture?

---

**Connections to Literature**
- Makridakis (2018) -- complexity doesn't guarantee better forecasting,
  confirmed directly here
- Fama (1970) -- EMH: uniform feature importance is consistent with
  no feature having persistent predictive power
- Fischer & Krauss (2018) -- found RF outperformed LSTM in trading
  returns despite LSTM's complexity. Our RF result here is consistent
  with their finding that RF is competitive but not dominant.


 ### (2026-06-29) Part 12: LSTM (Long Short-Term Memory)

**What I did:**
Built a 2-layer LSTM (64 units → 32 units → Dense sigmoid
output) with Dropout(0.2) for regularization. Lookback window: 20 days.
3-fold walk-forward validation. EarlyStopping(patience=15).
Total trainable parameters: 31,393.

**Environment issue encountered (significant):**
TensorFlow failed to install/run on my main Python 3.11 environment.
Two separate incompatible problems stacked on top of each other:
  1. The newest TensorFlow (2.12+) requires AVX2 CPU instructions.
     Unfortunately my CPU does not support AVX2 which was confirmed via DLL load failure
  2. The oldest TensorFlow that does NOT require AVX2 (2.10.0 and
     earlier) was never compiled for Python 3.11 so pip returned
     "No matching distribution found for tensorflow==2.10.0" when
     I tried installing it on Python 3.11, the version I had.
  These two constraints don't overlap on my machine's Python version --
  there is no single TensorFlow version that is both AVX2-free and
  Python 3.11-compatible.

Fix: Installed Python 3.10.11 alongside my existing 3.11 (does not
replace it). Had to create an isolated virtual environment (tf_env) using
py -3.10 -m venv tf_env specifically for this script, so the rest of
my project keeps using my normal Python 3.11 setup untouched. Inside
tf_env, installed tensorflow==2.10.0, pandas, numpy, scikit-learn,
matplotlib.

Second issue inside tf_env: numpy installed at its newest version
(2.x) by default since I didn't pin it. However, TensorFlow 2.10.0's compiled
C extensions were built against numpy 1.x's internal API and crashed
with "_ARRAY_API not found" / "numpy.core._multiarray_umath failed
to import" when numpy 2.x was present. Fixed by downgrading the numpy version by:
pip install "numpy<2.0" inside tf_env.

Lesson: deep learning libraries have much stricter and narrower
hardware + dependency requirements than classical/statistical libraries.
Every model up through Part 11 (ARIMA, GARCH, GBM, Logistic, Ridge,
RF, XGBoost) ran without any environment issues on my normal setup.
LSTM was the first model in the entire project complex enough to
require its own isolated environment. This is worth noting in
my Methodology or Limitations section -- it's a real practical
consideration for anyone trying to reproduce deep learning results,
not just a personal inconvenience.


**Results (mean across 3 folds):**
  Fold 1: 17 epochs, Accuracy=56.38%, F1=0.7156
  Fold 2: 17 epochs, Accuracy=50.19%, F1=0.6425
  Fold 3: 32 epochs, Accuracy=56.64%, F1=0.7205
  Mean Direction Accuracy: 54.40%
  Mean F1 Score:           0.6928
  vs Naive Baseline:       +0.47pp
  Mean Epochs Run:         22.0

**What I learned:**
LSTM landed on the EXACT SAME mean accuracy (54.40%)
and EXACT SAME margin over baseline (+0.47pp) as Logistic Regression
from Part 10 -- despite having 31,393 parameters and reading 20-day
sequences vs Logistic's 9 simple coefficients and zero memory.
This directly reproduces the Fischer & Krauss (2018) finding that
more model complexity does not reliably improve financial forecasting.
Fold 2 essentially collapsed to baseline (50.19%) -- regime-dependence,
same pattern seen in ARIMA and Logistic Regression.

### (2026-06-30) - Part 13 GRU (Gated Recurrent Unit)

**What I did:**
Built a 2-layer GRU (64 units → 32 units → Dense sigmoid
output) with Dropout(0.2) for regularization. Architecture mirrors
Part 12 LSTM exactly — only GRU layers substituted for LSTM layers —
to isolate the effect of architecture specifically.
Lookback window: 20 days. 3-fold walk-forward validation.
EarlyStopping(patience=15, restore_best_weights=True).
Total trainable parameters: 23,841 (vs LSTM's 31,393 — 24% fewer).

**Results (mean across 3 folds):**

  Fold 1: 16 epochs, Accuracy=55.12%, F1=0.7083
  Fold 2: 20 epochs, Accuracy=51.83%, F1=0.6735
  Fold 3: 37 epochs, Accuracy=57.02%, F1=0.7079
  Mean Direction Accuracy: 54.66%
  Mean F1 Score:           0.6966
  vs Naive Baseline:       +0.73pp
  vs LSTM (Part 12):       +0.26pp
  Mean Epochs Run:         24.3

**What I learned:**
GRU marginally outperformed LSTM (54.66% vs 54.40%, +0.26pp) despite
having 24% fewer parameters. Both models landed within 1pp of each
other and within 1pp of Logistic Regression (54.40%). This is the
central finding of the deep learning phase: three structurally
different models — a 9-coefficient linear classifier, a 31,393-
parameter LSTM, and a 23,841-parameter GRU — all converge on
essentially the same accuracy (~54-55%). The performance ceiling
is in the DATA, not the architecture. The signal in daily S&P 500
direction is too weak for any sequential deep learning model to
exploit beyond what a simple linear model already captures.

Fold 2 again collapsed toward baseline (51.83%) — the same
regime-dependence pattern seen across ARIMA, Logistic Regression,
and LSTM. All three deep learning folds map to the same time
periods as Part 12, confirming this is a market property during
that period, not a model flaw.

GRU trained slightly faster than LSTM (24.3 mean epochs vs 22.0)
and converged more smoothly — visible in the training loss curve.
This is consistent with GRU's simpler gate structure requiring
fewer updates to reach a stable solution.

**Problems:** 
None. Ran in tf_env (Python 3.10, tensorflow==2.10.0,
numpy<2.0) — same environment as Part 12, no new setup required.