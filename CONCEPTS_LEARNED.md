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

## Machine Learning

**(Fischer & Krauss (2018) key finding:** LSTM achieved only 56% directional accuracy on S&P 500 constituents. Random Forest outperformed LSTM in trading returns despite deep learning's reputation for superiority. This is a published benchmark to compare our own results against.

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

### Classical Models
Models that existed before machine learning, built on mathematical and statistical theory rather than learning from data patterns. These form the "classical" side of the three-way comparison in this project.

**ARIMA (AutoRegressive Integrated Moving Average):** the standard classical statistical benchmark for time-series forecasting. 
Three components:
  - AR (AutoRegressive): predicts using weighted past returns."Auto" means it regresses on itself -- past values of the same series.
  - I (Integrated): differencing to achieve stationarity. Set to 0 here since returns are already stationary from Day 2.
  - MA (Moving Average): uses past forecast ERRORS (not past returns) to correct future predictions.
  Equation: r(t) = c + φ1*r(t-1) + θ1*ε(t-1) + ε(t) Order notation: ARIMA(p, d, q) -- p=AR lags, d=differencing, q=MA lags. We used ARIMA(1,0,1).