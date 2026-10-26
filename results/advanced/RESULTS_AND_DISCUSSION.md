# Paper 2 — Automatically generated results snapshot

## Forecast definition and split
Forecast origin is after the close on date t; target is log(Close[t+1])-log(Close[t]).
Rows: 11418 training, 3806 validation, 3807 untouched test observations.
Test period: 2010-07-09 to 2025-08-28. AR order and MLP settings were selected using validation only.

## Held-out evidence
Best RMSE among tested candidates: training_mean (RMSE 0.01093841; MAE 0.007175698; R-squared against test mean -0.0003).
Full-feature MLP RMSE: 0.01120137. Graph-masked MLP RMSE: 0.01096869. Random-mask MLP RMSE: 0.01096634.
The graph-masked model's directional accuracy is 0.517 with 100.0% prediction coverage; directional accuracy is descriptive and is not evidence of statistical significance by itself.

## Evidence-based conclusion
The tested graph-masked MLP does not establish a reliable forecasting improvement. Its RMSE is 0.01096869, versus 0.01120137 for the unrestricted MLP and 0.01096634 for the random-mask ablation. For the graph-masked versus full-feature squared-error comparison, the primary HAC test gives p=0.1073 and Holm-adjusted p=0.3497; this does not reject equal expected loss at the conventional 5% level. The training_mean baseline has the lowest held-out RMSE among candidates tested. Accordingly, results support a cautious negative/null conclusion: the present experiments do not demonstrate that the Granger-screened feature mask improves next-day S&P 500 return forecasts. They do not establish that graph-informed methods cannot help under other specifications or datasets.

## Interpretation and limitations
The graph screen is estimated only from training data, with Benjamini-Hochberg adjustment across features. FDR-significant selected features: 7. Graph-model mode: FDR-significant feature mask. When no feature passes the adjusted threshold, the one-feature fallback is exploratory and must not be interpreted as a discovered graph edge. The screen tests Granger predictive dependence, not structural causal effects. Forecast comparisons use squared-error loss differentials with a Bartlett/Newey-West HAC variance estimate, finite-sample n/(n-1) correction, and a t(n-1) reference. The primary bandwidth is 10; sensitivity results at bandwidths 0, 5, 10, and 20 are saved separately, with Holm adjustment within each comparison family. These tests are approximate and bandwidth-sensitive inference should be interpreted cautiously. Rolling-origin graph selection is re-estimated from each fold's prior history only; hyperparameters remain fixed from initial validation. Rolling-origin and calendar-regime results are robustness descriptions, not causal estimates. Negative out-of-sample R-squared values for most candidates indicate they do not improve upon the test-mean reference under that metric. Regime-specific differences are descriptive and should not be treated as evidence of a robust regime interaction without an appropriate inferential test.

## Outputs
test_metrics.csv; test_forecasts.csv; training_graph_edges.csv; neural_validation_tuning.csv; rolling_origin_metrics.csv; rolling_origin_graph_screens.csv; regime_metrics.csv; forecast_comparison_tests.json; forecast_comparison_hac_sensitivity.json; three 300-dpi figures.
