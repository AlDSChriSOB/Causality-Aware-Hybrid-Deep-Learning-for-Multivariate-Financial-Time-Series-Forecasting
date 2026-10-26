# Paper 2 — Final validated results and conclusion

## Forecast definition and evaluation design
The forecasting target is the next-trading-day S&P 500 log return, defined as `log(Close[t+1]) - log(Close[t])`. The data are divided chronologically into 11,418 training observations, 3,806 validation observations, and 3,807 untouched test observations. Hyperparameters are selected using chronological training/validation data only; final estimators are refit on the pre-test history and evaluated once on the held-out test period (2010-07-09 to 2025-08-28).

The Granger screen is estimated without access to test observations. Rolling-origin graph selection is re-estimated from the history available before each forecast block. The graph represents lagged predictive dependence and is not interpreted as proof of structural causality.

## Final graph-informed experiment
The genuine graph message-passing model uses 122 directed training-only graph edges among 14 input variables. Graph information changes the model input through directed, normalized incoming messages rather than merely masking features. Two matched controls are evaluated: a no-graph zero-message architecture and a random-graph architecture with the same number of edges and the same model family.

On the untouched 3,807-observation test set:

| Model | RMSE | MAE | R² vs test mean |
|---|---:|---:|---:|
| Graph message passing | 0.01119991 | 0.00744418 | -0.04871 |
| No-graph zero-message | 0.01131990 | 0.00741144 | -0.07130 |
| Random-graph ablation | 0.01120288 | 0.00740995 | -0.04926 |

The graph message-passing model therefore improves RMSE relative to the no-graph architecture, but its RMSE is essentially indistinguishable from the random-graph control. The random graph also has slightly lower MAE. The negative R² values indicate that none of these graph architectures improves on the test-mean reference under that metric.

## Paired statistical comparison
The final paired comparisons use the identical saved test predictions in `test_forecasts.csv`. Squared-error loss differences are evaluated with a Bartlett/Newey-West HAC variance estimator at the primary bandwidth of 10, a finite-sample correction, and a Student-t reference distribution. Holm adjustment is applied across the four predeclared graph-model comparisons.

- Graph message passing vs no-graph: mean loss difference A−B = -2.7021e-06, p = 0.5316, Holm-adjusted p = 1.0000.
- Graph message passing vs random graph: mean loss difference A−B = -6.6418e-08, p = 0.9732, Holm-adjusted p = 1.0000.
- Graph message passing vs training-mean baseline: mean loss difference A−B = 5.7892e-06, p = 0.1134, Holm-adjusted p = 0.4538.
- Graph message passing vs zero-return baseline: mean loss difference A−B = 5.6099e-06, p = 0.1231, Holm-adjusted p = 0.4538.

Positive A−B means model A has higher squared-error loss. None of the four comparisons rejects equal expected loss at the conventional 5% level after Holm correction. The graph-vs-random comparison is especially important: the graph topology itself does not provide evidence of a forecasting advantage in this experiment.

## Final conclusion
The evidence does **not** support the claim that the proposed Granger-informed graph architecture improves next-day S&P 500 return forecasting. Although genuine graph message passing reduces RMSE relative to the matched no-graph architecture (0.01119991 vs 0.01131990), this improvement is not statistically significant (Holm-adjusted p = 1.0000) and disappears when compared with a random graph of identical size (Holm-adjusted p = 1.0000). The training-mean baseline also remains competitive, with the earlier model-selection experiment identifying it as the lowest-RMSE benchmark among the tested candidate models.

Accordingly, the appropriate scientific conclusion is a **null/negative result for graph-specific forecasting improvement under the present design**. The experiments demonstrate that the proposed graph mechanism can be implemented without leakage and can alter the model representation, but they do not establish that the Granger-derived topology contains predictive information that improves out-of-sample return forecasts. This is a substantive result rather than a failed experiment: it shows that a statistically screened predictive-dependence graph should not automatically be assumed to provide useful forecasting structure when embedded in a neural forecasting model.

The result should not be interpreted as evidence that graph-based or causality-informed forecasting is ineffective in general. It is specific to this dataset, target, feature construction, graph-estimation procedure, message-passing design, forecast horizon and evaluation protocol. Granger predictive dependence is also distinct from structural causal effect, so the results cannot establish or refute economic causality.

## Robustness and limitations
The rolling-origin procedure re-estimates graph selection using only information available before each forecast block, reducing temporal leakage risk. Regime-specific and rolling-origin results should be treated as robustness descriptions rather than causal estimates. Directional accuracy is descriptive and should not be interpreted as evidence of statistical forecasting skill without a corresponding inferential framework.

The HAC inference is approximate and should be accompanied by bandwidth sensitivity results at 0, 5, 10 and 20 lags. The final graph experiment uses a fixed architecture and frozen graph topology after the initial training graph estimation; therefore, future work should examine alternative graph estimators, dynamic graphs, longer temporal contexts, stronger multivariate sequence architectures, and genuinely causal discovery methods before making broader claims.

## Reproducibility outputs
`test_forecasts.csv` contains the identical saved predictions used for the four paired comparisons. `forecast_comparison_hac.json` contains the primary HAC statistics and Holm-adjusted p-values. `test_metrics.csv`, `feature_graph_edges.csv`, `tuning.json`, rolling-origin outputs and regime outputs provide the remaining experiment record.
