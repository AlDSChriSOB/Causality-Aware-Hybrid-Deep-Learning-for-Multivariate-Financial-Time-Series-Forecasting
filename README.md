# Causality-Aware Hybrid Deep Learning for Multivariate Financial Time-Series Forecasting

## Overview

This repository contains the implementation and experimental framework for a rigorous evaluation of whether a Granger-informed graph representation can improve next-trading-day S&P 500 return forecasting.

The study is deliberately framed as a validation study rather than as a demonstration of guaranteed causal-model superiority. The central question is whether a graph constructed from statistically screened lagged predictive dependence provides forecasting information beyond an otherwise matched neural architecture and a topology-matched random-graph control.

## Research question

**Does a leakage-aware, Granger-informed graph provide statistically supported out-of-sample forecasting gains when incorporated into a neural message-passing architecture for next-day S&P 500 returns?**

## Key methodological contributions

- Chronological training, validation and untouched test evaluation.
- Training-only Granger graph estimation to reduce temporal leakage.
- Rolling-origin re-estimation of graph selection using only information available before each forecast block.
- Validation-only neural hyperparameter selection.
- A genuine graph message-passing representation in which graph topology changes the model input rather than merely masking features.
- Matched no-graph and random-graph ablations.
- Paired forecast-comparison tests using Bartlett/Newey–West HAC inference and Holm correction.
- Reproducible prediction files and statistical-test outputs generated from the same held-out test predictions.

## Main finding

The final graph message-passing model uses 122 directed training-only edges among 14 input variables. On 3,807 untouched test observations, it achieves RMSE 0.01120, compared with 0.01132 for the matched no-graph architecture and 0.01120 for the random-graph control.

The graph model therefore improves numerically over the no-graph architecture, but the improvement is not statistically significant and is not distinguishable from a random graph with the same number of edges. The paired HAC comparison gives Holm-adjusted p = 1.0000 for both graph-versus-no-graph and graph-versus-random comparisons.

**Conclusion:** the experiments do not establish graph-specific forecasting improvement under the present design. The result is a null/negative finding about this graph mechanism, dataset, target and evaluation protocol—not a claim that graph-based or causality-informed forecasting is ineffective in general.

## Evaluation design

The forecasting target is the next-trading-day S&P 500 log return:

`log(Close[t+1]) - log(Close[t])`

The data are divided chronologically into 11,418 training observations, 3,806 validation observations and 3,807 untouched test observations. Hyperparameters are selected using chronological training/validation data only. Final models are refit on the pre-test history and evaluated on the held-out test period.

The Granger graph represents lagged predictive dependence and is **not** treated as proof of structural economic causality. Graph selection is performed without access to future test observations.

## Statistical comparison

Four predeclared graph-model comparisons are evaluated from the identical saved test predictions:

1. Graph message passing vs no graph.
2. Graph message passing vs random graph.
3. Graph message passing vs training-mean baseline.
4. Graph message passing vs zero-return baseline.

The primary inference uses HAC bandwidth 10 with finite-sample correction and Holm adjustment across the comparison family. Sensitivity analyses are retained for bandwidths 0, 5, 10 and 20.

## Repository structure

```text
├── data/                   # Data loaders / data documentation
├── notebooks/              # Exploratory and original experiments
├── models/                 # Model implementations
├── causal/                 # Graph and causal utilities
├── utils/                  # Preprocessing and evaluation helpers
├── scripts/                # Reproducible experiment and validation scripts
├── results/advanced/       # Validated metrics, forecasts, tests and figures
├── PAPER_2_MANUSCRIPT.md   # Publication-oriented manuscript presentation
├── PAPER_2_FINAL_CONCLUSION.md
├── PAPER_2_RESEARCH_DESIGN.md
├── README.md
└── requirements.txt
```

## Reproducibility outputs

The validated experiment records include:

- `test_forecasts.csv` — the held-out predictions used for forecast comparison.
- `test_metrics.csv` — model-level test metrics.
- `training_graph_edges.csv` / graph-edge outputs — training-only graph selection.
- `rolling_origin_metrics.csv` — expanding-window robustness evaluation.
- `rolling_origin_graph_screens.csv` — fold-specific graph screens.
- `regime_metrics.csv` — descriptive regime robustness results.
- `forecast_comparison_tests.json` — primary forecast-comparison statistics.
- `forecast_comparison_hac_sensitivity.json` — HAC bandwidth sensitivity.
- `forecast_comparison_hac.json` — graph-model paired HAC comparisons.

## Limitations

The results are conditional on the available variables, graph-estimation procedure, forecast horizon and neural architecture. Granger predictive dependence does not establish structural causation. The study also does not establish that alternative graph estimators, dynamic graphs, longer temporal contexts or other financial datasets would produce the same result.

## Author

Christian Somtoo Obiechina  
MSc Artificial Intelligence and Data Science  
Research interests: Causal AI, Financial Machine Learning, Time-Series Forecasting
