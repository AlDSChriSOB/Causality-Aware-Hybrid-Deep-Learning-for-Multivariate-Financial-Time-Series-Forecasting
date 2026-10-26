# Paper 2 — Automatically generated results snapshot

## Forecast definition and split
Forecast origin is after the close on date t; target is log(Close[t+1])-log(Close[t]).
Rows: 11418 training, 3806 validation, 3807 untouched test observations.
Test period: 2010-07-09 to 2025-08-28. AR order and MLP settings were selected using validation only.

## Held-out evidence
Best RMSE among tested candidates: training_mean (RMSE 0.01093841; MAE 0.007175698; R-squared against test mean -0.0003).
Full-feature MLP RMSE: 0.01120137. Graph-masked MLP RMSE: 0.01096869.
Consult test_metrics.csv, test_forecasts.csv and forecast_comparison_tests.json for the complete measured results.

## Interpretation
The graph screen is estimated only from training data, with Benjamini-Hochberg adjustment across features. FDR-significant selected features: 7. Graph-model mode: FDR-significant feature mask. When no feature passes the adjusted threshold, the one-feature fallback is exploratory and must not be interpreted as a discovered graph edge. The screen tests Granger predictive dependence, not structural causal effects. Forecast comparisons use squared-error loss differentials with a Bartlett/Newey-West HAC variance estimate, finite-sample n/(n-1) correction, and a t(n-1) reference. The primary bandwidth is 10; sensitivity results at bandwidths 0, 5, 10, and 20 are saved separately, with Holm adjustment within each comparison family. These tests are approximate and bandwidth-sensitive inference should be interpreted cautiously. Rolling-origin graph selection is re-estimated from each fold's prior history only; hyperparameters remain fixed from initial validation. Rolling-origin and calendar-regime results are robustness descriptions, not causal estimates. Directional accuracy excludes exactly zero predictions and reports coverage separately.

## Outputs
test_metrics.csv; test_forecasts.csv; training_graph_edges.csv; neural_validation_tuning.csv; rolling_origin_metrics.csv; rolling_origin_graph_screens.csv; regime_metrics.csv; forecast_comparison_tests.json; forecast_comparison_hac_sensitivity.json; three 300-dpi figures.
