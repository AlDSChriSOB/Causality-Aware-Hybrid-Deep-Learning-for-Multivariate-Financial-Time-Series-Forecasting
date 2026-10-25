# Paper 2 Research Design

## Working title

**Causality-Informed Hybrid Deep Learning for S&P 500 Return Forecasting Under Market Regime Changes**

The existing repository title, *Causality-Aware Hybrid Deep Learning for Multivariate Financial Time Series Forecasting*, can remain the project title. The paper should use “causality-informed” unless the study establishes stronger causal identification than predictive temporal dependence. A DAG or Granger-causality test alone does not establish intervention-level causation.

## 1. Research problem and motivation

Financial forecasting models can exploit correlations that are unstable across time. Their apparent performance can also be inflated by random data splits, preprocessing leakage, in-sample predictions, or metrics that are unsuitable for returns. This study will test whether a transparent, temporally justified structure learned from past market observations improves out-of-sample forecasts and robustness compared with otherwise comparable unrestricted models.

The current repository contains daily S&P 500 OHLCV data (Date, Open, High, Low, Close, Volume) and a notebook with log returns, moving averages, RSI, volume changes, Granger tests, an LSTM, and an attention model. The design below is intended to correct the notebook's evaluation weaknesses before any performance claims are made.

## 2. Aim

To evaluate whether incorporating lagged, statistically supported dependence structure into a hybrid sequence model improves out-of-sample S&P 500 return forecasting, particularly across changing volatility regimes.

## 3. Research questions

**RQ1.** Do lagged price, range, momentum, and trading-volume features provide reproducible predictive information about next-day S&P 500 returns beyond a simple historical benchmark?

**RQ2.** Does a causality-informed LSTM–attention model improve out-of-sample forecast accuracy and directional performance relative to statistical, machine-learning, and unrestricted deep-learning baselines?

**RQ3.** Are any gains stable across calm and high-volatility periods, forecast horizons, and rolling evaluation windows?

**RQ4.** Does the proposed structure improve interpretability or robustness enough to justify its additional estimation and computational cost?

### Hypotheses

- **H1:** Some lagged features will provide incremental predictive information after controlling for multiple testing and accounting for temporal dependence; this is an empirical hypothesis, not an assumed result.
- **H2:** The causality-informed model will reduce out-of-sample MAE or RMSE relative to the matched unrestricted LSTM–attention model.
- **H3:** Any benefit will vary by market regime and forecast horizon rather than necessarily holding uniformly.
- **H4:** Restricting connections using a learned graph will reduce model complexity or instability without materially degrading predictive performance.

All hypotheses may be rejected. The paper must report negative and null results.

## 4. Data and unit of analysis

### Primary dataset

Use the repository's SP500.csv, after verifying its source, licensing, frequency, date coverage, duplicate dates, missingness, zero-volume records, and any changes in data construction. Record the dataset's final start and end dates and the number of observations after every cleaning step.

The unit of analysis is one trading day. The initial study uses only variables available in the current OHLCV dataset. Do not describe it as a macro-financial multivariate study unless additional, properly aligned data are actually added.

### Data quality checks

1. Parse dates, sort ascending, enforce one observation per trading date, and inspect missing, duplicate, non-positive, and implausible values.
2. Inspect the early historical period separately. Flat or repeated OHLC values can reflect historical recording limitations; do not assume the entire sample is equally informative.
3. Report sensitivity analyses for a defensible modern-data subsample and the longest reliable sample. Define the cutoff from documented data quality, not from which period produces better results.
4. Do not interpolate market prices across non-trading days. Treat exchange holidays as absent observations rather than missing daily market records.
5. Freeze the source file and record a checksum or version identifier for reproducibility.

## 5. Target and feature definitions

### Primary target

Use the **next-trading-day log return**:

\[
r_{t+1}=\log(C_{t+1})-\log(C_t)
\]

where C_t is the closing level on day t. The forecast is made after day t's information is available and predicts the return from close t to close t+1.

A secondary target may be the cumulative five-trading-day log return, defined consistently as \(\log(C_{t+5})-\log(C_t)\). The five-day task must be evaluated separately and must not be presented as five independent daily forecasts.

### Candidate predictors

All predictors must be computed using data available no later than the forecast origin. Use lagged values as model inputs.

- Lagged log returns: 1, 2, 5, 10, and 20 trading days.
- High–low range scaled by close: (H_t-L_t)/C_t.
- Open-to-close return: log(C_t/O_t).
- Overnight gap: log(O_t/C_{t-1}).
- Log-volume change: log(V_t)-log(V_{t-1}), only where volume is positive and the field is reliable.
- Rolling volatility of returns over 5-, 20-, and 60-day windows.
- RSI and moving-average-derived features, provided their formulas and windows are documented.

Moving averages and RSI are deterministic transformations of past prices, not independent causal drivers. Their interpretation must reflect this. Avoid redundant feature proliferation and pre-register a primary feature set.

## 6. Causal and temporal structure

### Operational definition

In this paper, “causality-informed” means the model uses an explicit graph of **lagged predictive dependencies**, estimated without future observations and interpreted under stated assumptions. It does not mean that observational market data alone identify structural causal effects.

### Procedure

1. Transform price levels into returns or other justified stationary representations before causal analysis. Check stationarity using complementary diagnostics and inspect structural breaks; do not rely on one ADF result alone.
2. Estimate lagged relationships on the training segment only. Candidate approaches are Granger tests and a time-series causal-discovery method such as PCMCI, with the conditional-independence test and assumptions stated.
3. Set a maximum lag before testing (initial proposal: 5 trading days for the primary graph); report sensitivity to alternative lag limits.
4. Correct for multiple hypothesis tests, for example using Benjamini–Hochberg false-discovery-rate control. Report the selected threshold and effect sizes, not p-values alone.
5. Build a directed lagged graph from training-only results. The graph must distinguish lagged edges from contemporaneous relationships. Do not treat a contemporaneous correlation between derived indicators as evidence of temporal causation.
6. Freeze the graph for each evaluation fold, or re-estimate it using only data available at that fold's training cutoff. Never use the test period to choose graph edges, features, thresholds, or hyperparameters.
7. Compare the graph with a domain-informed pre-specified graph and with an unrestricted model. A causal mask should be applied to the model's connections in a documented, reproducible way.

### Important caveat

Volume may not Granger-predict returns in this dataset. If the tests do not support the proposed edge, remove it or retain it only in a clearly labelled theory-driven sensitivity analysis. Do not describe an unsupported edge as empirically confirmed.

## 7. Model specification

### Benchmarks

Use the same target, forecast origins, and test dates for every model.

1. **Zero-return forecast:** predicts a next-day return of zero.
2. **Historical-mean forecast:** mean return estimated from the training segment only.
3. **Regularised linear model:** ridge regression on the engineered lagged features.
4. **Econometric benchmark:** AR model or ARIMA selected using training data only.
5. **Tree-based benchmark:** gradient-boosted trees or random forest on the same lagged feature set.
6. **Vanilla LSTM:** sequence model without attention or graph restrictions.
7. **LSTM–attention:** the current hybrid architecture without graph masking.
8. **Causality-informed LSTM–attention:** the proposed model with a training-only lagged dependency graph or mask.

GRU and Transformer models may be added if computational resources permit, but should not displace the core matched comparisons above. The key comparison is between the same architecture with and without the causal-structure constraint.

### Sequence construction

Use a fixed lookback window, initially 60 trading days, as a pre-specified default. Construct each input sequence from observations ending at time t and pair it with the target return at t+1. Confirm this indexing with a small hand-checked example and unit tests.

### Training controls

- Fit imputers, scalers, feature selectors, and other learned preprocessing steps on training data only.
- Apply the fitted transformations to validation and test data without refitting.
- Select architecture and hyperparameters using the validation segment or inner rolling validation only.
- Use early stopping based on validation loss.
- Fix and report random seeds; run multiple seeds for neural models and report mean and dispersion.
- Document software versions, parameter counts, training duration, and hardware where relevant.
- Keep the final test set untouched until model and evaluation choices are locked.

## 8. Evaluation protocol

### Primary split

Use chronological evaluation, never a random train/test split. A starting design is 60% training, 20% validation, and 20% final test, ordered by date. Report exact date boundaries and sample counts. Because financial regimes change, supplement this split with expanding-window or rolling-origin evaluation.

At each forecast origin, all features, transformations, graph estimates, and model parameters must use only data that would have been available then. For the strongest design, conduct nested rolling-origin evaluation: inner chronological validation for model selection and outer folds for unbiased comparison.

### Metrics

Report:

- MAE and RMSE for return forecasts.
- Out-of-sample R-squared relative to a stated benchmark, including negative values where applicable.
- Directional accuracy, with a clearly specified treatment of zero returns and ties.
- A sign-based baseline and a confusion table for up/down direction where useful.
- Calibration or error-distribution diagnostics where appropriate.
- A simple transaction-cost-aware strategy result only as a secondary analysis, with turnover, trading assumptions, risk-free rate, costs, and exposure explicitly stated.

Do **not** use MAPE as a primary metric for returns: returns can be zero or close to zero, making percentage errors undefined or unstable. Do not calculate headline performance from predictions on the same observations used to fit the model.

### Statistical comparison

Compare forecast losses on the same test dates. Use a Diebold–Mariano test with an appropriate variance estimator for serial dependence and overlapping multi-step forecasts; explain the loss differential and any small-sample correction. Supplement p-values with confidence intervals or block-bootstrap estimates of the performance difference. Adjust for multiple model/horizon comparisons or designate a single primary comparison in advance.

## 9. Regime robustness and ablation

Define regimes without looking at test outcomes. One reproducible option is a rolling realised-volatility measure and a threshold estimated on training data, with sensitivity checks using an alternative threshold. Label high- and low-volatility observations according to that frozen rule.

Required analyses:

1. Primary causal mask versus no mask, with architecture and training budget held constant.
2. LSTM versus LSTM–attention.
3. Volume features included versus excluded.
4. Graph estimated from training data versus a pre-specified theory-driven graph.
5. Alternative lookback windows, such as 20 and 60 days.
6. Reliable modern-data sample versus longer historical sample.
7. Performance by volatility regime and by rolling test window.
8. Seed sensitivity and model complexity.

The analysis should make clear whether improvements arise from graph constraints, attention, extra features, or tuning differences.

## 10. Interpretability and reproducibility

Visualise the lagged graph with edge direction, lag, effect estimate, and uncertainty or adjusted significance clearly identified. For attention weights, avoid claiming that they are causal explanations. Use them as descriptive model diagnostics and, where feasible, compare them with feature perturbation or ablation results.

Publish or retain the data-cleaning code, feature definitions, split dates, seeds, hyperparameter search space, graph-estimation settings, and scripts that regenerate tables and figures. Respect data licensing. Include a model card or limitations statement covering non-stationarity, unobserved confounding, proxy features, historical data quality, and limited generalisability from one index.

## 11. Threats to validity

- **Leakage:** full-sample scaling, graph estimation on all dates, or in-sample predictions can inflate results.
- **Weak identification:** Granger tests and PCMCI results depend on assumptions, conditioning sets, lag limits, and data quality; predictive temporal dependence is not proof of intervention-level causation.
- **Regime instability:** relationships learned in one period may not transfer to another.
- **Single-market scope:** the S&P 500 alone cannot establish cross-market generality.
- **Feature redundancy:** moving averages and RSI are derived from prices and may add little independent information.
- **Metric misuse:** MAPE is unsuitable when the target can be zero or near zero.
- **Multiple comparisons:** searching many features, models, and horizons can produce apparently significant results by chance.

## 12. Decision rules and expected contribution

The main claim will be supported only if the causality-informed model improves the pre-specified out-of-sample metric against the matched unrestricted model and the difference is robust enough to survive the planned statistical and regime checks. If not, report the null or negative result and investigate whether the graph adds value in interpretability or stability rather than accuracy.

The intended contribution is a reproducible test of whether explicit lagged-dependency constraints improve financial sequence forecasting under temporal distribution shift—not a claim that the method guarantees profitable trading or discovers the true causal structure of markets.

## 13. Planned paper structure

1. Introduction and contributions
2. Related work: financial forecasting, causal time-series analysis, and structured deep learning
3. Data and feature construction
4. Causal-structure estimation
5. Forecasting models and experimental protocol
6. Results: overall, regime-specific, and ablation
7. Robustness and limitations
8. Conclusion and reproducibility statement

## 14. Immediate implementation checklist

- [ ] Verify dataset provenance, coverage, schema, and reliable-period cutoff.
- [ ] Replace in-sample predictions with chronological out-of-sample predictions.
- [ ] Move scaling and all learned preprocessing inside each training fold.
- [ ] Confirm target alignment and sequence indexing.
- [ ] Re-run causal tests using training data only and correct for multiple testing.
- [ ] Implement the zero-return, linear/econometric, tree, LSTM, LSTM–attention, and graph-constrained comparisons.
- [ ] Remove MAPE from primary reporting.
- [ ] Implement rolling-origin evaluation, uncertainty estimates, and regime analysis.
- [ ] Regenerate all tables and figures from saved predictions.
- [ ] Only then write results and abstract claims.

**Status:** This document defines the proposed research design. It does not claim that the planned experiments have been rerun or that the hypotheses have been supported.
