## Week 1 Plan (Data + Foundations)

**Target/asset decision (locked in):** S&P 500 index (^GSPC), daily data,
2000-present. Primary target: next-day direction (up/down). Secondary
target: next-day return (for regression models).

**Why these choices:** Index data avoids single-company noise and is the
standard benchmark in the forecasting literature. Direction prediction is
more forgiving to evaluate for a first project; return prediction adds
depth, so we keep both.

### (2026-06-14) Day 1: Data Loading and Exploration
- What I did: Downloaded S&P 500 daily price data from 2000 to present (6641 rows). i did this by importing yfinance into my Visual Studio python code. During calling the data I found a bug where the yfinance returned data in MultiIndex columns so I had to flatten it using pandas. I used the pandas' pct_change() function to compute daily returns from the price data. Used matplotlib to generate and review plots of price, returns, and a histogram of returns. Saved raw and processed data as CSVs using pandas.
- What I learned: Raw price trends upward over decades with no fixed "level" (non-stationary), while daily returns hover around zero the whole time (stationary). This is why financial models work with returns instead of raw price. Also saw that the histograp of returns has fatter tails than a normal distribution would predict (confirmed numerically using pandas" built-in skew() and kurt() functions). 
- Problems: Had a path error where the script couldn't find the saved CSV depending on which folder the terminal was running from -- fixed using Python's OS library to make the script locate files relative to its own location instead of the current working directory.


### (2026-06-15) Day 2: Stationarity and Autocorrelation
- What I did: Used the statsmodels library's adfuller() function to run the Augmented Dickey-Fuller (ADF) test on price and on returns. Used statsmodels' plot_acf() and plot_pacf() funcitons to generate autocorrelation plots for returns, amd separately for squared returns (computed using basic Python exponentiation, Return**2).
- What I learned: The ADF test on price gave a p-value of 1.0000 (no evidence of stationarity). The ADF test on returns gave a p-value of about 0.000000 (strong evidence of stationarity). The ACF/PACF plots on raw returns showed almost no statistically meaningful autocorrelation at any lag, meaning past returns don't reliably  predict the direction of future returns. The ACF/PACF plots on squared returns showed strong, statistically meaninful autocorrelation at every lag tested (around 0.40 at lag 2), meaning past volatility. This is called volatility clustering.
- Problems: None once the path issue from Day 1 was fixed (same fix carried over using the os library).

### (2026-06-16) Day 3: Feature Engineering
- What I did: Used pandas' shift() and rolling() functions to build engineered features -- lagged returns (1,2,3, and 5 days back), rolling averages (5 day and 10-day), and rolling volatility (5 day, 10-day, and 20-day standard deviation of returns). Defined two prediction targets using shift(-1) target_direction (1 if tomorrow's return is positive, 0 otherwise) and target_return (tomorrow's actual return value). Dropped rows with missing values created by the lag/rolling windows. Saved the result as sp500_features.csv.
- What I learned: Why features must only use past data (lookahead bias/data leakage) --shift(1) pulls past data (safe for features), while shift(-1) pulls future data (only safe for defining the target, since that's literally the answer we're trying to predict, not an input). Verified this manually: return_lag_1 on any row exactly matches Return from the row before it, and target_return on any row exactly matches Return from the row after it -- confirming the shifts work correctly in both directions.
- Results: Final dataset has 6619 rows and 17 columns after dropping 21 rows with missing values. Class balance for target_direction: 53.8% up-days vs 46.2% down-days -- mildly imbalanced, meaning a model that always predicts "up" would get ~53.8% accuracy without learning anything real. This becomes the baseline every real model needs to beat by a meaningful, statistically significant margin.
- Problems: None -- ran cleanly once the file was saved into the correct folder.

### (2026-06-17) Day 4: Walk-Forward Train/Test Split (complete)
- What I did: First tested the WRONG way to split time-series data --  randomly shuffling rows before splitting into train/test (the default behavior of sklearn's train_test_split). Then did it the RIGHT way: a single chronological split where all training data comes before all test data, with no overlap. Then built a more rigorous version called walk-forward validation using sklearn's TimeSeriesSplit, which repeats the chronological split 5 times with an expanding training window, giving 5 separate train/test pairs (folds) instead of just one.
  
- What I learned: When data is randomly shuffled, both the resulting "training" pile and "testing" pile end up spanning the entire 26-year date range -- confirmed directly in my own output (both showed date ranges from 2000 to 2026). This means a model could secretly train on recent data (e.g. 2024) and get "tested" on older data (e.g. 2010), which is impossible in real life and makes the evaluation meaningless. The chronological split fixes this: training data always comes before test data in time, so the model is never trained on the future and tested on the past -- only ever the other way around. Walk-forward validation repeats this idea 5 times, each fold's test period starting exactly one day after that fold's training period ends, giving a more trustworthy evaluation than relying on just one split.
  
- Results: Chronological 80/20 split: train = 2000-02-02 to 2021-02-18 (5295 rows), test = 2021-02-19 to 2026-05-28 (1324 rows), no overlap. 5-fold walk-forward validation confirmed correct: e.g. Fold 3 train ends 2013-04-02, Fold 3 test starts 2013-04-03 -- exactly adjacent, no gap, no overlap.
  
- Problems: Initially found the terminology (row, fold, train/test set) confusing when looking at the output all at once. Worked through it by focusing on one number at a time instead of the whole output -- e.g. comparing just the WRONG approach's two date ranges side by side first, before moving to the RIGHT approach.
