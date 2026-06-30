## Concepts Learned
This file's purpose is to show the different concepts I learned througout this project.

Organized into four sections: Python Basics, Financial Forecasting Terms, Machine Learning, and Math/Statistics. New entries get added to the relevant section as the project progresses.

## Python Basics 
**import:** loads a toolkit (library) into your program so you can use its tools "as" gives it a short nickname so you don't retype the full name every time (e.g. import pandas (a library) as pd)

**function call:** using a tool form a toolkit by writing its name with parentheses, e.g. yf.downlaod(ticker, start="2000-01-01", end="2026-06-01"). The things inside the parentheses are inputs you're handing the tool: what comes back gets stored in a variable.

**DataFrame:** a table of data with rows and columns, like a programmable spreadsheet, provided by the pandas library.data.head() shows the first 5 rows;data.shape shows (number of rows, number of columns).

**if statement:** a line of code that only runs the code below it when a condition is True. Used in Day 1 to check isinstance(data.columns, pd.MultiIndex) -- only flatten the columns if they're actually in that complicated two-layer format.

**.shift(n):** slides a column's values up or down by n rows. shift(1) pushes values down 1 row, so each row now shows YESTERDAY's value. shift(-1) pushes values up 1 row, so each row now shows TOMORROW's value. This is the core mechanic behind avoiding lookahead bias: features use shift(1) (past only), targets use shift(-1) (defining the future answer).

**.rolling(window=n):** looks at the last n rows at a time (a "moving window") and lets you calculate something across just those rows, like .mean() or .std(), e.g. a 5-day rolling average or rolling volatility.2. 

Virtual environment (venv): an isolated, self-contained copy of Python
and its installed packages, separate from your main system Python.
Created with `py -X.X -m venv env_name`. Lets you install conflicting
package versions for different projects/scripts without one breaking
the other. Activated with env_name\Scripts\activate (Windows).

Dependency version pinning: explicitly specifying a package's version
(e.g. tensorflow==2.10.0 or "numpy<2.0") instead of letting pip install
the latest by default. Necessary when an older library was compiled
against an older version of a dependency's internal API and breaks
against newer versions of that same dependency.

Binary incompatibility: when two pieces of compiled code expect
different internal data layouts or APIs from each other, even though
they're nominally "compatible" by version number. Not a normal code
bug -- a structural mismatch at the compiled level, not fixable by
editing your own Python script.

## Financial Forecasting Terms

**Return:** the percentage change in price from one day to the next, (today's price - yesterday's price) / yesterday's price. Used instead of raw price because returns are stationary and raw price isn't.

**Stationarity:** A stationary series has a roughly constant average and spread over time -- it wanders but keeps returning to about the same range, like a thermostat-controlled room. A non-stationary series drifts with no fixed "typical level," like the total mileage on a road trip.

  - Why it mattered: S&P 500 raw price is non-stationary (climbed from ~1400 in 2000 to 7500+ now, no snap-back tendency). Daily returns are stationary (hover near zero the whole time). This is why finance models almost always use returns, not raw price.
  - What confused me: I initially explained this backwards -- I said price has a high p-value BECAUSE it has no snap-back. Actually it's the reverse: the lack of snap-back behavior is what CAUSES the high p-value. The data's real-world behavior is the cause; the p-value is just a measurement of it.


**Volatility:** how much a price or return jumps around -- the size of the swings, regardless of direction. A +3% day and a -3% day are both "high volatility," even though one is a gain and one is a loss.

**Volatility clustering:** Big price swings tend to cluster near other big swings, and calm periods cluster near other calm periods -- markets alternate between turbulent and quiet stretches rather than mixing
randomly day to day.

  - Why it mattered here: confirmed statistically (not just by eye) via the squared-returns autocorrelation result below. Direct motivation for including GARCH as one of the classical models later, since GARCH is specifically built to forecast volatility using this clustering pattern.

 **Fat tails:** financial returns have more extreme days (big gains/losses) than a normal bell-curve distribution would predict. Confirmed numerically using skewness and kurtosis on the Day 1 returns histogram.

**Lookahead bias / data leakage:** Accidentally letting a model "see" future information when building a feature meant to predict that future. If a feature for "today" secretly includes tomorrow's actual data, results in testing will look great but be meaningless in real life, since that information wouldn't exist yet at prediction time.

  - Why it mattered: Every FEATURE column uses .shift(1) or similar (pulling from the past). Only the TARGET columns (target_direction, target_return) use .shift(-1) (pulling tomorrow's value), because defining the answer we want to predict is the one legitimate place to reference the future.

  **Efficient Market Hypothesis (EMH) -- Fama (1970):** in an efficient market, past prices contain no exploitable information about future prices. This is the theoretical reason financial forecasting is hard for any model, and why the Day 2 ACF/PACF showing near-zero autocorrelation was expected, not surprising.

**Geometric Brownian Motion (GBM) -- Samuelson (1965):** the mathematical model of stock price dynamics. Prices follow the equation dS = μS dt + σS dW, where μ is drift, σ is volatility, and dW is random noise. GBM is the mathematical embodiment of the EMH -- if prices truly follow GBM, no model should consistently beat it.

**Anti-momentum:** when the persistence baseline scores below 50%, it means the market has a slight tendency to reverse direction day-to-day rather than continue. Confirmed in Day 6 (49.77%). Consistent with near-zero ACF and the EMH.

**Naive baseline:** the simplest possible "model" -- a dumb rule requiring no learning. Used to establish the accuracy floor every real model must beat. Two variants: Always Up (53.93%) and Persistence (49.77%).

**GARCH as a volatility model, not a direction model:**
GARCH's real output is a variance forecast -- how much the market is expected
to move tomorrow, not which way. The direction accuracy number (49.23%) came
from a heuristic imposed on top of it: high predicted vol → predict down,
low predicted vol → predict up. That is not what GARCH was built for.
The right metric for GARCH in this project is volatility RMSE: 0.3601%.
That is what goes in the paper as the GARCH contribution.

**GARCH regime sensitivity:**
GARCH adapts slowly to sudden volatility regime changes. Fold 3 (2018–2021)
RMSE spiked to 0.50% vs ~0.30% in normal folds because of the COVID crash
in March 2020 -- a massive sudden spike the model had never seen in training.
This is a known limitation discussed in the volatility modeling literature.

**GBM as the mathematical finance tradition:**
GBM represents a completely different philosophy from ARIMA and GARCH.
Statistical models learn patterns from past data. GBM does not learn --
it says prices follow drift + randomness and simulates from that assumption.
The only things calibrated from data are μ (mean daily return) and σ
(std of daily returns). Everything else is pure mathematical theory.
This is why GBM is the third tradition in the research question -- it is
not statistical, not ML, but mathematical finance.

**Drift sensitivity:**
GBM direction forecasts are highly sensitive to the calibrated μ (drift).
When μ is estimated from a long bull market history, GBM predicts "up"
more often. In bull market test periods this works well (Fold 5: 57.19%).
In choppy or bear periods it fails badly (Fold 4: 45.69%). This is a
fundamental limitation -- GBM assumes drift is constant, which it is not.

**Phase 2 finding:**
No classical model beat the naive baseline on average. GBM came closest
at -1.83pp, ARIMA at -3.20pp, GARCH at -4.70pp. This is consistent with
EMH (Fama 1970) and Makridakis (2018). The ML models now need to beat
53.93% to justify their added complexity.

**First ML model to beat the naive baseline:**
Logistic Regression achieved 54.40% -- the first model in the entire
project to exceed the 53.93% naive floor. Small margin (+0.47pp) but
real. This is the turning point in the project narrative: simple ML
can extract a tiny signal that classical models could not.

**Why Ridge underperformed for direction:**
Ridge is optimized to minimize return magnitude error (RMSE), not
direction accuracy. Taking the sign of a predicted return to get a
direction is a weak heuristic -- the model was never trained to care
about getting the direction right, only about getting the size right.
This is why Logistic Regression (trained directly on direction) beat
Ridge for the direction task even though both are linear models.

**Anti-momentum signal confirmed in ML:**
return_ma_5 had the strongest negative coefficient -- recent upward
5-day momentum slightly predicts a down day. This is consistent with
the near-zero autocorrelation from Part 2 and the persistence baseline
scoring below 50% in Part 6. The signal is tiny but present.

**Uniform feature importance = EMH confirmation:**
Both RF and XGBoost spread importance nearly evenly across all 9 features.
No single feature dominated. This is exactly what the Efficient Market
Hypothesis predicts -- no past price feature should have persistent
strong predictive power. If one did, the market would arbitrage it away.

**Phase 3 finding:**
Only Logistic Regression beat the naive baseline across all ML models.
Adding non-linearity (RF, XGBoost) did not help and sometimes hurt.
The project narrative going into LSTM: if even ensemble methods with
200 trees can't beat a linear model, does deep learning change anything?

## Machine Learning

**(Fischer & Krauss (2018) key finding:** LSTM achieved only 56% directional accuracy on S&P 500 constituents. Random Forest outperformed LSTM in trading returns despite deep learning's reputation for superiority. This is a published benchmark to compare our own results against.

**Logistic Regression:**
Despite the name it is a classifier not a regressor. Takes features and
finds a LINEAR boundary separating up days from down days in feature space.
Output is a probability between 0 and 1 -- if above 0.5 predict up, else down.
Core equation: prediction = w1*x1 + w2*x2 + ... + wn*xn + b
Where x = features, w = learned weights, b = bias term.
C=0.1 means moderate regularization -- smaller C = stronger penalty.

**Ridge Regression:**
Linear regression with L2 regularization penalty added to prevent overfitting.
Predicts continuous return values, not direction directly.
Direction derived by taking the sign of the predicted return (positive = up).
alpha=1.0 controls regularization strength -- larger alpha = stronger penalty.
Real metric for Ridge is RMSE (0.010454), not direction accuracy.

**Regularization:**
Without it, linear models overfit by assigning huge weights to certain
features. Regularization adds a penalty term that keeps all weights small
and stable. Ridge uses L2 penalty = alpha * sum(w²). Logistic uses C
where smaller C = stronger regularization (inverse relationship).
Prevents the model from memorizing noise in training data.

**StandardScaler:**
Transforms each feature to mean=0 and std=1 before fitting.
Critical for linear models because features have very different scales
(return_lag_1 ≈ 0.001 vs volatility_20 ≈ 0.015). Without scaling the
model gives more weight to larger-valued features just because of scale,
not actual predictive power.
IMPORTANT: fit scaler ONLY on training data, then apply to test.
Fitting on test data leaks future information -- same lookahead bias issue
from Part 3 features, just appearing at a different stage.

**Feature coefficients (importance):**
In logistic regression, each feature gets a weight (coefficient).
Larger absolute value = more influence on the prediction.
Positive coefficient = feature pushes prediction toward "up."
Negative coefficient = feature pushes prediction toward "down."
return_ma_5 had the strongest coefficient (-0.044) -- recent upward
momentum slightly predicts reversal (anti-momentum, consistent with Day 2).
All volatility features had negative coefficients -- high vol predicts down.

**Random Forest:**
An ensemble of many decision trees, each trained on a random subset of
data rows and a random subset of features. Final prediction = majority
vote across all trees. The randomness makes each tree different so their
errors don't all go in the same direction -- when averaged out, errors
cancel and signal remains. This is called bagging (bootstrap aggregating).
Key parameters used:
- n_estimators=200: number of trees
- max_depth=4: limits tree depth to prevent overfitting
- min_samples_leaf=20: each leaf needs at least 20 samples
- max_features="sqrt": each tree only sees sqrt(9) ≈ 3 features

**XGBoost (Extreme Gradient Boosting):**
Builds decision trees SEQUENTIALLY -- each new tree is specifically
trained to correct the mistakes the previous trees made. This is called
gradient boosting. More powerful than Random Forest in theory but more
sensitive to hyperparameter tuning and more prone to overfitting on
noisy data.
Key parameters used:
- n_estimators=200: number of boosting rounds
- max_depth=3: shallower than RF -- boosting compensates for depth
- learning_rate=0.05: how much each tree corrects the previous
- subsample=0.8: each tree trained on 80% of rows
- colsample_bytree=0.8: each tree sees 80% of features

**Bagging vs Boosting:**
Bagging (Random Forest): trees built independently in parallel,
errors averaged out. More robust to noisy data.
Boosting (XGBoost): trees built sequentially, each fixing prior errors.
More powerful on clean data but more sensitive to noise.
On financial data (very noisy), bagging tends to be more stable --
consistent with RF outperforming XGBoost here.

**Why tree models don't need StandardScaler:**
Decision trees split on feature value thresholds (e.g. "is return_lag_1
> 0.005?"). The absolute scale of the feature doesn't change which
threshold is chosen -- only the relative ordering of values matters.
Scaling would not hurt but is completely unnecessary for tree models.
This is different from linear models where scale directly affects weights.

**Feature importance (tree models):**
Random Forest: measures average reduction in impurity (Gini) across
all splits using each feature across all trees.
XGBoost: measures average gain in accuracy from splits using each feature.
Both are on different scales so they can't be directly compared as numbers,
but the ranking of features can be compared.
Both models showed nearly uniform importance across all 9 features --
no single feature dominated. Consistent with EMH.

**Why complex ML underperformed simple ML here:**
1. Features are weak -- near-zero autocorrelation means barely any
   signal exists regardless of model complexity.
2. RF and XGBoost have more capacity to memorize noise in training data
   (overfit) than Logistic Regression's constrained linear boundary.
3. Consistent with Makridakis (2018): complexity does not automatically
   help on noisy financial time-series data.

   **Sliding window / sequence creation:**
converting a flat time series intooverlapping windows of fixed length, where each window becomes one
training example. Used exclusively by LSTM in this project -- every
other model uses single-day feature vectors instead.

**return_sequences (LSTM parameter):** controls whether an LSTM layer
outputs its full sequence of hidden states (one per timestep) or just
the final hidden state. True is needed when stacking another sequence-
processing layer on top. False is used right before a Dense layer that
needs one summary vector, not a sequence.

**LSTM parameter count:**
scales with 4x the gates (forget, input, output,candidate state) multiplied by the relationship between input size and
hidden units. A 64-unit LSTM layer with 9 input features has 18,944
parameters, while Logistic Regression with the same 9 features has
just 10 (9 weights + 1 intercept). LSTM's complexity comes from this
gate structure, not just "more layers."

**restore_best_weights:**
an EarlyStopping setting that rewinds the model to its best-performing epoch (lowest validation loss) rather than
keeping whatever the model looked like at the moment training stopped.

**Complexity-performance tie:**
when a simple model and a far more complexmodel converge on the same accuracy, it suggests the exploitable signal
in the data is small enough that added model capacity has nothing
extra to find -- both models are converging on the same weak signal,
not failing independently.

## Math / Statistics

**p-value (for the ADF test):** A number from a statistical test indicating how likely a result could be due to random chance. Below 0.05 = strong evidence against "this is just chance" (here: strong evidence the series IS stationary).

  - Why it mattered here: ADF test on price gave p = 1.0000 (no evidence of stationarity at all). ADF test on returns gave p ~ 0.000000 (overwhelming evidence of stationarity). Confirmed what the plots suggested, with an actual statistical test instead of just eyeballing a chart.

**Autocorrelation and lag:** Autocorrelation asks whether a series correlates with a past version of ITSELF (not a different variable). Lag = how far back: lag 1 = yesterday, lag 2 = two days ago, etc.

  - Why it mattered here: ACF/PACF on raw returns showed almost every bar inside the "could be random noise" band -- meaning past returns barely predict future returns (direction is hard to predict). ACF/PACF on SQUARED returns showed bars clearly outside that band at every lag tested (e.g. ~0.40 at lag 2) -- meaning past volatility (size of moves) DOES predict future volatility.

  - What confused me: I needed several passes to connect "0.40" (strength of relationship) with "outside the band" (statistically real, not noise) into one complete idea, instead of treating them as two separate facts.

 **Walk-forward validation (train/test splitting for time-series):** The correct way to split time-ordered data into training and testing sets. Training data must always come entirely BEFORE testing data in time -- never shuffled, never mixed. Walk-forward validation repeats this chronological split multiple times (called "folds"), each time with a bigger training window, sliding forward through time.

- Why it mattered here: Randomly shuffling rows before splitting (the default behavior in most ML libraries) lets a model train on data from later years and get tested on data from earlier years -- which is impossible in real-world forecasting, since you'd never actually have future information available at the time you're making a past prediction. This is the same lookahead bias problem from Day 3's features, just appearing at the train/test-split level instead. Confirmed directly: a shuffled split produced a "train" date range and a "test" date range that both spanned the entire 2000-2026 period, proving they were mixed together in time, not separated.

- What confused me: The sheer number of new terms at once (row, fold, train set, test set) made the output hard to parse initially. It helped to compare just two specific numbers side by side at a time, rather than trying to absorb the whole printed output in one pass.

**Confusion matrix:** a table showing the four possible outcomes of a binary prediction -- true positives, true negatives, false positives, false negatives. The Always Up confusion matrix had zero predictions in the "down" column, confirming it never predicts down regardless of what actually happened.

**F1 score:** a metric that balances precision (of all "up" predictions, how many were correct?) and recall (of all actual "up" days, how many did we catch?). More informative than accuracy alone when classes are imbalanced. Always Up got F1 = 0.70 because it catches every true "up" day but misses every "down" day entirely.

**ARCH/GARCH origin -- Engle (1982), Bollerslev (1986):** GARCH models time-varying volatility. The variance of returns is not constant -- it depends on past squared errors (past shocks). GARCH(1,1) using just one lag proved sufficient for most financial series and remains the standard volatility benchmark. Directly motivated by the squared-returns ACF result from Day 2.

**RMSE (Root Mean Squared Error):** measures how far predicted return values are from actual values. Penalizes large errors more heavily because of the squaring. Lower = better.

**MAE (Mean Absolute Error):** average of absolute differences between predicted and actual return values. Less sensitive to outliers than RMSE. Lower = better.

**Regime-dependence:** when a model performs very differently across time periods. ARIMA got 44% in 2008-2013 but 57% in 2017-2022 -- suggesting it picks up patterns that only exist in certain market conditions, not a stable generalizable signal. (More to be added: standard deviation/skewness/kurtosis detail, walk-forward validation logic, statistical significance testing for comparing models -- coming in Phase 1 Day 4 and Phase 5.)

**Heteroskedasticity:**
Variance is not constant over time. Some periods are calm, some are chaotic.
Confirmed in Day 2 -- squared returns ACF showed significant spikes, meaning
big moves cluster together. GARCH was built specifically to model this.

**The GARCH(1,1) Equation:**
σ²(t) = ω + α·ε²(t−1) + β·σ²(t−1)

σ²(t)     = today's variance (what we're predicting)
ω         = long-run baseline variance (constant floor)
ε²(t−1)   = yesterday's squared shock (how surprising yesterday was)
σ²(t−1)   = yesterday's variance (how volatile it already was)

**α + β ≈ 1 in financial data:**
In most real GARCH fits on financial returns, α + β is very close to 1.
This means volatility is highly persistent -- a high-volatility period tends
to stay high for a long time before mean-reverting back to ω.
If α + β = 1 exactly, the model is called IGARCH (integrated GARCH).

**The GBM Equation:**
dS = μS dt + σS dW

S   = current price
μ   = drift (average daily upward tendency)
σ   = volatility (size of random moves)
dt  = one time step (one day)
dW  = random noise drawn from a normal distribution

The discrete one-step version used in code:
S(t+1) = S(t) * exp((μ - 0.5σ²)dt + σ√dt * Z)

Where Z ~ N(0,1) is a standard normal random draw.

**Ito's Lemma correction -- the (μ - 0.5σ²) term:**
Without this correction, the expected log return would be biased upward.
The 0.5σ² term corrects for the mathematical asymmetry introduced by
taking the exponential of a normally distributed variable. This comes
from Ito's Lemma in stochastic calculus. It is subtle but critical --
leaving it out would produce systematically overoptimistic forecasts.

**Monte Carlo simulation:**
Instead of solving the GBM equation once, simulate it N=1000 times,
each time drawing different random noise Z. This gives 1000 possible
futures. The median of those 1000 simulated prices is the point forecast.
The spread of the distribution shows uncertainty -- wide spread means
high uncertainty, narrow spread means low uncertainty.

**Random seed (np.random.seed(42)):**
Setting a fixed seed before Monte Carlo simulation makes results
reproducible. Anyone who runs the same code gets the same 1000 random
draws. Without it, results would differ slightly every run.

**Cross-model regime fragility:**
when structurally different models (ARIMA, Logistic Regression, LSTM) all show their weakest performance
during the same real-world time period, the weakness is likely a
property of the market during that period, not a flaw specific to any
one model's architecture. A stronger finding than any single model's
fold-by-fold variance alone.

### Classical Models
Models that existed before machine learning, built on mathematical and statistical theory rather than learning from data patterns. These form the "classical" side of the three-way comparison in this project.

**ARIMA (AutoRegressive Integrated Moving Average):** the standard classical statistical benchmark for time-series forecasting. 
Three components:
  - AR (AutoRegressive): predicts using weighted past returns."Auto" means it regresses on itself -- past values of the same series.
  - I (Integrated): differencing to achieve stationarity. Set to 0 here since returns are already stationary from Day 2.
  - MA (Moving Average): uses past forecast ERRORS (not past returns) to correct future predictions.
  Equation: r(t) = c + φ1*r(t-1) + θ1*ε(t-1) + ε(t) Order notation: ARIMA(p, d, q) -- p=AR lags, d=differencing, q=MA lags. We used ARIMA(1,0,1).
  
  **GARCH (Generalized AutoRegressive Conditional Heteroskedasticity):**
A classical statistical model that predicts time-varying volatility -- how
uncertain tomorrow will be based on how uncertain it has recently been.
It does NOT predict direction. It predicts variance.

**ARCH vs GARCH:**
ARCH (Engle 1982) only uses past squared shocks to predict variance.
GARCH (Bollerslev 1986) adds a lagged variance term -- making it more
flexible and persistent. GARCH(1,1) nests ARCH(1) as a special case
when β = 0. GARCH(1,1) is now the standard volatility model in finance.

**GARCH(1,1) notation:**
p=1 means one lag of the squared shock term (α)
q=1 means one lag of the variance term (β)
GARCH(1,1) is almost always sufficient for daily financial returns.

**GBM + Monte Carlo:**
Geometric Brownian Motion is the mathematical model of stock price dynamics.
First formalized by Samuelson (1965), it became the foundation of
Black-Scholes (1973) and all of modern mathematical finance.
GBM assumes prices follow a random walk with drift -- consistent with
the Efficient Market Hypothesis. It does not learn from patterns in data.

**Why GBM beat ARIMA and GARCH for direction:**
GBM outperformed both statistical classical models (52.10% vs 50.73%
and 49.23%) despite fitting nothing to the data beyond μ and σ.
This suggests that the statistical structure ARIMA and GARCH try to
exploit in past returns either does not exist or is too weak to help --
consistent with the near-zero ACF from Day 2 and the EMH.

**Connection to Black-Scholes:**
The mathematical exposition (second artifact) will derive the
Black-Scholes PDE from GBM assumptions. Day 9 is the practical
implementation of the same foundation that derivation sits on.

