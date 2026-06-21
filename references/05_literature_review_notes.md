# Literature Review Notes

This file logs every paper read for the research project, organized by
theme. Each entry includes: full citation, what the paper did, key
finding, and how it relates to my own project. This becomes the backbone
of the paper's Introduction and Literature Review sections.

Format:
- Citation
- What they did
- Key finding
- Relevance to my project

---

## Theme 1: ML vs. Classical Methods -- Broad Comparisons

### Makridakis et al. (2018)
**Citation:** Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018).
Statistical and Machine Learning forecasting methods: Concerns and ways
forward. PLOS ONE.

**What they did:** Evaluated 8 classical statistical methods against 10
machine learning methods across 1,045 monthly time series from the M3
forecasting competition. One of the most cited comparative studies in
this space.

**Key finding:** Classical statistical methods (especially simpler ones
like exponential smoothing) generally outperformed or matched ML methods
on these time series. ML methods did not consistently demonstrate
superiority over classical ones despite their complexity.

**Relevance to my project:** Directly relevant as a baseline expectation --
this is one of the most cited papers arguing that complexity doesn't
always help. My results should be discussed in light of this finding,
whether they confirm or contradict it.

---

### Springer Nature Review (2025)
**Citation:** Journal of Big Data, "Artificial intelligence and classical
statistical models for time series forecasting: a comprehensive review."
Springer Nature, December 2025.

**What they did:** Meta-analysis of over 150 studies comparing deep
learning approaches (LSTM, GRU, Transformers) against classical
statistical models (ARIMA, etc.) for time series forecasting.

**Key finding:** Deep learning approaches, particularly those using Adam
and RMSProp optimizers, improve forecasting accuracy by up to 14%
compared to traditional methods. Hybrid models showed superior
performance in multi-step predictions and handling volatility.

**Relevance to my project:** Provides a broad, recent benchmark for how
much improvement deep learning (LSTM in my Phase 4) can be expected to
provide over classical methods. The "up to 14%" figure is useful context
-- not always, and not unconditionally.

---

### PMC Review -- Statistical Models vs. Machine Learning (2025)
**Citation:** "Selected Topics in Time Series Forecasting: Statistical
Models vs. Machine Learning." MDPI Entropy, March 2025.

**What they did:** Compared ML methods (Random Forest, gradient boosting,
LSTM, Transformers, CNN, TCN) against classical parametric models
(ARIMA, GARCH). Also analyzed results from Makridakis forecasting
competitions M1-M6.

**Key finding:** Classical parametric models perform well in linear
settings, but ML methods handle nonlinear patterns better. GARCH-type
modeling remains valuable specifically for volatility/probability
forecasting -- not return direction prediction. ML methods capture more
of the probabilistic structure of data beyond just covariance.

**Relevance to my project:** Directly relevant to my GARCH vs. ML
comparison -- confirms GARCH's niche (volatility, not direction) and
that ML's advantage specifically appears in nonlinear regime detection.

---

## Theme 2: ARIMA vs. LSTM on S&P 500 Specifically

### Pilla & Mekonen (2025)
**Citation:** Pilla, P., & Mekonen, R. (2025). Forecasting S&P 500 Using
LSTM Models. arXiv:2501.17366.

**What they did:** Compared ARIMA and LSTM models on S&P 500 prediction
using historical price data and technical indicators. Evaluated using
MAE and RMSE.

**Key finding:** ARIMA achieved MAE of 462.1, RMSE of 614, accuracy
89.8%. LSTM without additional features outperformed both, achieving MAE
of 175.9, RMSE of 207.34, accuracy 96.41%. LSTM captured both short- and
long-term dependencies better than ARIMA.

**Relevance to my project:** Same index (S&P 500), same two model types
(ARIMA vs. LSTM) as two of my core models. Note: their accuracy figures
look suspiciously high (96.41%) -- likely because they evaluated on
price level, not return direction. This is a methodological concern to
flag in my own paper (predicting price level is easier than predicting
direction or return, and less meaningful for a trading strategy).

---

### Comparative Study (2023)
**Citation:** "Evaluating ARIMA and LSTM Approaches for Predicting S&P
500 Index Movements: A Comparative Analysis." SCITEPRESS, 2023.

**What they did:** Compared ARIMA and LSTM on S&P 500 closing price
prediction.

**Key finding:** LSTM demonstrated lower prediction errors than ARIMA on
the S&P 500, confirming that deep learning captures dependencies
classical models miss.

**Relevance to my project:** Another direct S&P 500 comparison. Same
concern about price-level vs. return/direction evaluation applies.

---

### Transformer vs. Classical (2026)
**Citation:** "A Comparative Study of Transformer-Based and Classical
Models for Financial Time-Series Forecasting." Journal of Risk and
Financial Management, March 2026.

**What they did:** One-day-ahead return forecasting (not price level)
comparing classical baselines (ARIMA, Random Forest) with deep learning
models including Transformer variants.

**Key finding:** Return predictability is expected to be limited under
the Efficient Market Hypothesis (Fama, 1970), and any exploitable
structure is typically weak and unstable. Confirmed in their results --
even advanced models showed limited edge on one-day-ahead return
forecasting.

**Relevance to my project:** This paper is the most methodologically
similar to mine -- one-day-ahead RETURN forecasting (not price level),
with EMH as the theoretical backdrop. Their finding of limited
predictability aligns with my Day 2 ACF/PACF result (past returns show
almost no autocorrelation). Strongly worth citing in my Discussion
section.

---

## Theme 3: Random Forest and XGBoost vs. Classical Methods

### Random Forest vs. ARIMA (2023)
**Citation:** "Comparison of Stock Price Prediction in Context of ARIMA
and Random Forest Models." BCP Business & Management, 2023.

**What they did:** Compared ARIMA and Random Forest models on stock price
prediction.

**Key finding:** ARIMA is more suitable for short-term forecasting but
limited to linear relationships. Random Forest has higher accuracy but
is more complex and computationally expensive. Neither dominates
unconditionally.

**Relevance to my project:** Directly relevant to Phase 3 (Random Forest
vs. ARIMA in my lineup). The "neither dominates unconditionally" finding
is consistent with the broader literature and likely to appear in my
results too.

---

### XGBoost vs. OLS vs. Random Forest (2022)
**Citation:** "The Comparison of Stock Return Prediction for Random Forest,
Ordinary Least Square, and XGBoost." BCP Business & Management, 2022.

**What they did:** Compared XGBoost, Random Forest, and OLS (linear
regression) for stock return prediction using only technical indicators.

**Key finding:** XGBoost and Random Forest both outperformed OLS across
all tested stocks. Difference between RF and XGBoost was subtle. Choice
of stocks affected model performance.

**Relevance to my project:** Supports including both Random Forest and
XGBoost in my comparison (Phase 3) and using OLS/Ridge as a linear
baseline. The "subtle difference between RF and XGBoost" finding
suggests I may find similar results, which is worth discussing.

---

## Theme 4: Efficient Market Hypothesis (EMH) -- Theoretical Backdrop

### Fama (1970)
**Citation:** Fama, E. F. (1970). Efficient Capital Markets: A Review of
Theory and Empirical Work. Journal of Finance, 25(2), 383-417.

**What they did:** Foundational paper defining and testing the Efficient
Market Hypothesis -- the idea that asset prices fully reflect all
available information, making consistent above-market returns impossible
through prediction.

**Key finding:** In an efficient market, past prices contain no
information useful for predicting future prices beyond what's already
reflected in the current price -- a "random walk."

**Relevance to my project:** This is the theoretical reason why financial
forecasting is hard and why my Day 2 ACF/PACF result (minimal
autocorrelation in raw returns) was expected. My whole research question
implicitly asks: do ML models find any edge that the EMH says shouldn't
exist? If they don't, that's not a failure -- it's a confirmation of a
foundational economic theory.

---

### Southampton University Study
**Citation:** "Bridging the divide in financial market forecasting:
machine learners vs. financial economists." University of Southampton
Institutional Repository.

**What they did:** Extensive forecasting simulation across 34 financial
indices over 6 years, explicitly trying to resolve the contradiction
between ML claims of high predictive accuracy and financial economists'
claims of market efficiency.

**Key finding:** The contradiction largely resolves when methodology is
carefully controlled -- many ML "wins" over classical methods in prior
papers disappear under proper evaluation (walk-forward validation,
realistic transaction costs, proper baselines).

**Relevance to my project:** Directly supports why our methodology
(walk-forward validation, naive baseline comparison, significance
testing) is critical. This paper is essentially arguing for everything
we built in Days 1-4, and should be cited in the Methodology section.

---

## Key Themes Emerging From Literature

1. ML does not consistently beat classical methods -- results depend
   heavily on methodology, asset class, and evaluation metric.
2. Predicting price LEVEL is much easier (and less meaningful) than
   predicting return DIRECTION -- many "impressive" ML papers are
   actually doing the easier task.
3. GARCH's niche is specifically volatility forecasting, not direction
   prediction -- consistent with my Day 2 squared-returns finding.
4. The Efficient Market Hypothesis provides the theoretical reason why
   one-day-ahead return forecasting is hard for any model.
5. Proper evaluation (walk-forward, naive baseline, significance
   testing) often eliminates apparent ML advantages that appear under
   simpler evaluation setups.

---

---

## Theme 5: Fischer & Krauss (2018) -- The Most Cited LSTM Finance Paper

### Fischer & Krauss (2018)
**Citation:** Fischer, T., & Krauss, C. (2018). Deep learning with long
short-term memory networks for financial market predictions. European
Journal of Operational Research, 270(2), 654-669.
https://doi.org/10.1016/j.ejor.2017.11.054

**What they did:** Applied LSTM networks to predict daily return
direction of all S&P 500 constituent stocks from 1992-2015.
Benchmarked against Random Forest, gradient-boosted trees, deep neural
networks, and logistic regression.

**Key finding:** LSTM achieved 56% directional accuracy on daily
predictions -- a modest but statistically meaningful edge over the
naive 50% baseline. Critically, Random Forest (0.43% daily return) and
gradient-boosted trees (0.37%) actually outperformed deep neural
networks (0.33%) in terms of trading returns, despite deep learning's
reputation for superiority. Performance was highly sensitive to dataset
size, feature quality, and signal-to-noise ratio.

**Relevance to my project:** This is one of the most cited papers
connecting LSTM to S&P 500 prediction -- I should directly reference it
when discussing Phase 4 (LSTM) results. The finding that Random Forest
outperformed LSTM in trading returns is particularly important: it
suggests "more complex" doesn't always mean "better," which is exactly
my research question. The 56% directional accuracy baseline is a real,
published benchmark to compare my own LSTM results against.

---

## Theme 6: Foundational GARCH Papers (Classical Mathematical Baseline)

### Engle (1982) -- Original ARCH Paper
**Citation:** Engle, R.F. (1982). Autoregressive Conditional
Heteroskedasticity with Estimates of the Variance of United Kingdom
Inflation. Econometrica, 50(4), 987-1007.

**What they did:** Introduced the ARCH model -- the first formal
statistical framework for modeling time-varying volatility in financial
time series.

**Key finding:** The variance of financial returns is not constant
over time -- it depends on past squared errors (past shocks). This
formalized the volatility clustering phenomenon mathematically for the
first time.

**Relevance to my project:** This is the foundational paper for GARCH
(which generalizes ARCH). My Day 2 result (strong autocorrelation in
squared returns) is literally the empirical observation Engle's ARCH
model was built to capture. Should be cited in the GARCH/methodology
section of the paper.

---

### Bollerslev (1986) -- GARCH Generalization
**Citation:** Bollerslev, T. (1986). Generalized Autoregressive
Conditional Heteroskedasticity. Journal of Econometrics, 31(3),
307-327.

**What they did:** Extended Engle's ARCH model to GARCH, allowing
conditional variance to depend on both past squared errors AND past
conditional variances -- making it far more practical and parsimonious
for real financial data.

**Key finding:** GARCH(1,1) -- using just one lag of each -- proved
sufficient to capture volatility clustering in most financial time
series. It became the standard benchmark for volatility modeling in
quantitative finance and remains widely used today.

**Relevance to my project:** GARCH is one of my six core models.
This is its original paper. In my paper's Methodology section, I
should cite this and briefly explain the GARCH(1,1) equation, since
this project targets applied math rigor, not just library calls.

---

### GARCH on Global Stock Indexes Including S&P 500 (2024)
**Citation:** Marisetty, N. (2024). Applications of GARCH Models in
Forecasting Financial Market Volatility: Insights from Leading Global
Stock Indexes. SSRN Working Paper.

**What they did:** Applied several GARCH variants (GARCH, EGARCH,
GJR-GARCH, TGARCH) to major global indices including the S&P 500 over
a 20-year period (2004-2023), covering the 2008 crisis and COVID-19.

**Key finding:** GARCH models remain strong at capturing volatility
clustering across different market regimes. TGARCH performed best at
capturing asymmetric volatility (negative shocks cause larger
volatility than positive ones -- the "leverage effect").

**Relevance to my project:** Same index (S&P 500), overlapping time
period. Confirms GARCH is an appropriate classical model choice for
this specific asset. Also introduces the leverage effect as a
limitation of basic GARCH(1,1) -- worth mentioning in the Discussion
section as a refinement opportunity.

---

## Theme 7: Geometric Brownian Motion (GBM) -- Mathematical Baseline

### Samuelson (1965) -- GBM Foundation
**Citation:** Samuelson, P.A. (1965). Rational Theory of Warrant
Pricing. Industrial Management Review, 6(2), 13-39.

**What they did:** Formalized Geometric Brownian Motion (GBM) as the
standard mathematical model for stock price dynamics, building on
Bachelier's 1900 random walk and Osborne's 1959 empirical work.
Established that discounted stock prices follow a martingale -- meaning
no trading strategy based on past prices should consistently profit.

**Key finding:** Stock prices, modeled as GBM, follow a stochastic
differential equation: dS = μS dt + σS dW, where μ is drift (expected
growth), σ is volatility, and dW is a Wiener process (random noise).
This directly implies that log-returns are normally distributed -- a
testable assumption your Day 1 histogram already partly addressed (fat
tails suggest GBM's normality assumption is imperfect for real data).

**Relevance to my project:** GBM/Monte Carlo is now a core model in
my lineup (Phase 2, alongside ARIMA/GARCH). This is the foundational
paper. The GBM model is the mathematical embodiment of the Efficient
Market Hypothesis -- if GBM accurately describes prices, then no model
should meaningfully beat it. Its inclusion is what gives this project
the "mathematical methods" dimension in the research question, beyond
just "statistical methods."

---

### Black & Scholes (1973) -- GBM Applied to Option Pricing
**Citation:** Black, F., & Scholes, M. (1973). The Pricing of Options
and Corporate Liabilities. Journal of Political Economy, 81(3),
637-654.

**What they did:** Used GBM as the assumed price process to derive the
famous Black-Scholes formula for pricing European options -- one of
the most impactful papers in all of financial economics, earning
Scholes and Merton the 1997 Nobel Prize in Economics.

**Key finding:** If stock prices follow GBM, then the price of an
option can be derived analytically from just five inputs: current
price, strike price, time to expiry, risk-free rate, and volatility.
This made options pricing rigorous and spawned the modern derivatives
market.

**Relevance to my project:** Black-Scholes is built directly on GBM,
so understanding GBM (which is in our core model lineup) gives you the
mathematical foundation that Black-Scholes sits on. If the extension
goes into Black-Scholes territory, this is the primary citation. Even
without the extension, citing this in the GBM section shows awareness
of where this mathematical tradition leads.

---

## Key Themes Emerging From Literature (Updated)

1. ML does not consistently beat classical methods -- results depend
   heavily on methodology, asset class, and evaluation metric.
2. Predicting price LEVEL is much easier (and less meaningful) than
   predicting return DIRECTION -- many impressive-looking ML papers
   are actually doing the easier task.
3. GARCH's niche is specifically volatility forecasting, not direction
   prediction -- consistent with my Day 2 squared-returns ACF finding.
4. The Efficient Market Hypothesis (Fama 1970) and GBM (Samuelson
   1965) provide the theoretical reason why one-day-ahead return
   forecasting is hard for any model -- GBM/EMH is the mathematical
   null hypothesis my entire project is testing against.
5. Proper evaluation (walk-forward, naive baseline, significance
   testing) often eliminates apparent ML advantages -- the Southampton
   paper confirmed this directly.
6. Even Fischer & Krauss (2018), the most-cited LSTM finance paper,
   found Random Forest outperformed LSTM in trading returns -- "more
   complex" does not reliably mean "better," which is exactly what
   my project is designed to test rigorously.
7. GARCH originates from Engle (1982) and Bollerslev (1986) --
   foundational papers that should be cited when using GARCH, not
   just the library that implements it.
