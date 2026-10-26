# Does a Granger-Informed Graph Improve Financial Return Forecasting? A Leakage-Aware Evaluation with Topology-Matched Ablations

## Abstract

Graph- and causality-informed neural forecasting models are often motivated by the expectation that statistically identified dependencies provide useful structure for prediction. However, apparent gains can arise from temporal leakage, feature-selection effects, additional model capacity, or differences in representation rather than from the proposed graph topology itself. This study evaluates that distinction in next-trading-day S&P 500 log-return forecasting.

We develop a leakage-aware evaluation protocol in which graph selection is performed using training information only, neural hyperparameters are selected chronologically using a validation set, rolling-origin graph selection is re-estimated from historical information available before each forecast block, and the final evaluation is performed on an untouched test period. We implement genuine graph message passing so that the estimated directed graph changes the model representation rather than merely selecting features. The proposed architecture is compared with a matched no-graph model and a topology-matched random-graph ablation. Forecast differences are evaluated using paired squared-error loss comparisons with Bartlett/Newey–West HAC inference and Holm correction.

On 3,807 untouched test observations, the graph message-passing model obtains RMSE 0.01120, compared with 0.01132 for the no-graph architecture and 0.01120 for the random-graph control. The graph-versus-no-graph comparison is not statistically significant (Holm-adjusted p = 1.0000), and the graph-versus-random comparison is also not significant (Holm-adjusted p = 1.0000). Comparisons with the training-mean and zero-return baselines are likewise not significant after correction. The results therefore do not establish graph-specific forecasting improvement. Instead, they show that a plausible Granger-informed topology can alter a neural representation and produce a numerical improvement over a no-graph control without providing statistically supported predictive information beyond a topology-matched random graph. The study highlights the importance of leakage control, matched ablations and forecast-comparison inference when evaluating causal-information mechanisms in financial machine learning.

**Keywords:** financial time series, S&P 500, Granger predictive dependence, graph neural networks, causal machine learning, leakage control, forecast comparison, HAC inference, ablation study

## 1. Introduction

Financial return forecasting is difficult because financial series exhibit weak signal-to-noise ratios, non-stationarity, changing dependence structures and regime variation. Multivariate machine-learning models can exploit information across related variables, but unconstrained models may also learn unstable or spurious relationships. This has motivated the use of causal and graph-based representations intended to encode economically or statistically meaningful dependencies.

A central methodological difficulty is determining whether a proposed graph actually contributes predictive information. A graph-informed model can appear superior because graph construction inadvertently uses future information, because feature selection reduces dimensionality, because the graph increases effective model capacity, or because the particular neural representation is useful irrespective of the graph topology. A comparison against an unstructured model alone therefore does not isolate the contribution of the graph.

This study addresses that problem through a deliberately stringent evaluation design. The graph is estimated from training information only. Hyperparameters are selected using chronological validation data. Rolling-origin graph selection uses only the history available before each forecast block. The final test set remains untouched until model selection and graph construction are complete. Most importantly, the proposed graph message-passing model is compared with both a no-graph architecture and a random graph with the same number of edges and model family. The random topology provides a direct test of whether the specific statistically screened graph contains useful information beyond the representational effect of graph message passing.

The research question is therefore:

> **Does a leakage-aware Granger-informed graph provide statistically supported out-of-sample forecasting gains when incorporated into a neural message-passing architecture for next-day S&P 500 returns?**

The contribution is not the claim that causality-aware forecasting is universally beneficial. Rather, the study provides a reproducible evaluation framework for testing graph-specific forecasting claims and reports a rigorous null result when the proposed topology does not survive matched ablation and statistical comparison.

## 2. Methodology

### 2.1 Forecasting target and temporal design

The forecasting target is the next-trading-day S&P 500 log return:

`r_(t+1) = log(Close_(t+1)) - log(Close_t)`

The observations are divided chronologically into 11,418 training observations, 3,806 validation observations and 3,807 untouched test observations. No random shuffling is used for model selection. Neural hyperparameters are selected on the training/validation sequence, after which final estimators are refit using the pre-test history and evaluated once on the held-out test period.

### 2.2 Granger-informed graph construction

Directed edges are estimated using Granger predictive-dependence tests among the 14 input variables. Multiple-testing control is applied to the tested relationships. The resulting graph contains 122 directed training-only edges in the final graph-message-passing experiment.

The graph is interpreted strictly as a representation of lagged predictive dependence. A Granger relationship is not treated as evidence of structural economic causality, intervention effects or causal identification.

### 2.3 Leakage control

Graph construction is separated from the test period. The final test observations are not used to determine graph edges or neural hyperparameters. For rolling-origin evaluation, graph selection is repeated using only the history available before each forecast block. This prevents future observations from influencing the feature topology used for subsequent forecasts.

The evaluation therefore distinguishes three information stages:

1. **Training:** estimate graph structure and fit candidate models.
2. **Validation:** select neural hyperparameters and model configurations.
3. **Test:** evaluate the frozen design once on unseen observations.

### 2.4 Graph message-passing architecture

Unlike the earlier feature-masking formulation, the final architecture uses the selected directed graph to construct normalized incoming messages between variables. The graph therefore changes the model representation directly. The same neural model family is used across the graph and matched controls.

The principal comparison consists of:

- **Graph message passing:** training-only Granger-informed topology.
- **No-graph control:** identical architecture with graph messages removed.
- **Random-graph ablation:** identical architecture and edge count, but with a randomly generated topology.

The random-graph comparison is critical because it controls for the possibility that message passing or additional structured inputs help independently of the particular Granger-derived topology.

### 2.5 Forecast comparison

The final comparison uses the identical saved test predictions for all four predeclared graph-model comparisons. Squared-error loss differences are evaluated with a Bartlett/Newey–West HAC variance estimator using bandwidth 10 as the primary specification, finite-sample correction and a Student-t reference distribution. Holm adjustment controls the family-wise error rate across the predeclared comparisons.

Sensitivity analyses are retained for HAC bandwidths 0, 5, 10 and 20.

## 3. Results

### 3.1 Held-out forecasting performance

| Model | RMSE | MAE | R² vs test mean |
|---|---:|---:|---:|
| Graph message passing | 0.01119991 | 0.00744418 | -0.04871 |
| No-graph zero-message | 0.01131990 | 0.00741144 | -0.07130 |
| Random-graph ablation | 0.01120288 | 0.00740995 | -0.04926 |

The graph architecture has lower RMSE than the no-graph control. However, the random-graph ablation produces virtually the same RMSE and slightly lower MAE. Thus, the numerical improvement over the no-graph model cannot by itself be interpreted as evidence that the Granger-informed topology is useful.

The negative R² values also indicate that these graph architectures do not outperform the test-mean reference under the reported R² formulation.

### 3.2 Paired statistical comparisons

| Comparison | Mean loss difference A−B | Raw p | Holm-adjusted p |
|---|---:|---:|---:|
| Graph vs no graph | -2.7021 × 10⁻⁶ | 0.5316 | 1.0000 |
| Graph vs random graph | -6.6418 × 10⁻⁸ | 0.9732 | 1.0000 |
| Graph vs training mean | 5.7892 × 10⁻⁶ | 0.1134 | 0.4538 |
| Graph vs zero return | 5.6099 × 10⁻⁶ | 0.1231 | 0.4538 |

None of the predeclared comparisons rejects equal expected squared-error loss at the conventional 5% level after Holm correction. The graph-versus-random comparison is particularly informative because the two architectures differ primarily in topology while retaining the same edge count and model family.

### 3.3 Rolling-origin and regime robustness

Rolling-origin graph selection is re-estimated from information available before each forecast block. These evaluations are treated as robustness descriptions rather than independent claims of causal effect. Differences across blocks and market regimes do not provide sufficient evidence for a general graph-by-regime interaction without a dedicated inferential test.

### 3.4 Directional accuracy

Directional accuracy is reported descriptively and coverage is reported separately. It is not treated as evidence of statistically significant predictive skill without an appropriate directional-forecast inference procedure.

## 4. Discussion

The central result is that the Granger-informed graph does not provide statistically supported forecasting gains under the present experimental design. The graph message-passing model improves RMSE relative to removing graph messages, but the same improvement is reproduced by a random graph with the same number of edges. This sharply weakens the interpretation that the gain comes from the specific predictive-dependence structure identified by the Granger screen.

This distinction is important for causal and graph-based machine learning. A statistically plausible graph can be meaningful as a representation without being useful for forecasting. Conversely, a numerical improvement from a graph architecture does not establish that the graph contains economically or causally informative structure. The topology-matched ablation is therefore more informative than a simple comparison with an unconstrained baseline.

The result also illustrates why leakage control is essential. If graph selection were performed using the complete sample, the estimated topology could incorporate information unavailable at forecast time. The present design prevents this by estimating the final graph without access to the held-out test observations and by re-estimating graph selection chronologically in the rolling-origin analysis.

The findings should not be interpreted as evidence against graph-based forecasting generally. They are conditional on the dataset, 14-variable feature set, next-day return target, graph-estimation method, message-passing representation and evaluation period. Different variables, horizons, graph estimators, dynamic graph mechanisms or sequence architectures may produce different results.

## 5. Implications

For researchers, the results suggest that claims of causal or graph-informed forecasting benefit should be evaluated against topology-matched controls rather than only conventional unconstrained baselines. For practitioners, the findings provide no statistical justification for selecting the present Granger-informed graph architecture over simpler alternatives solely on forecasting-error grounds.

The negative result is itself informative because it separates three propositions that are often conflated:

1. A predictive-dependence graph can be estimated without leakage.
2. A graph can change the representation of a neural forecasting model.
3. The particular estimated topology provides statistically significant forecasting value.

The present experiments support the first two propositions but not the third.

## 6. Limitations and future work

The study uses one financial market, one forecast horizon and a fixed set of observed variables. Granger predictive dependence is not structural causal identification, and latent confounding or omitted variables may affect the estimated topology. The graph is also frozen after its initial estimation in the final graph-message-passing experiment.

Future work should evaluate dynamic and window-specific graphs, alternative causal-discovery methods, longer temporal contexts, stronger multivariate sequence architectures and independent financial datasets. Any future claim of graph-specific benefit should retain the same basic safeguards: chronological information flow, topology-matched controls and statistical comparison on genuinely untouched observations.

## 7. Conclusion

The study evaluated whether a leakage-aware Granger-informed graph improves next-day S&P 500 return forecasting through neural message passing. The final graph architecture reduced RMSE relative to a matched no-graph architecture, but its performance was essentially identical to that of a random graph with the same number of edges. None of the four predeclared paired HAC comparisons was statistically significant after Holm correction.

The final conclusion is therefore a **null result for graph-specific forecasting improvement under the present design**. The work demonstrates a rigorous and reproducible framework for testing such claims and shows that an apparently plausible predictive-dependence graph does not automatically translate into statistically supported out-of-sample forecasting gains.

## Suggested figures and captions

### Figure 1. Experimental information-flow design
**Caption:** Chronological training, validation and untouched test protocol. Graph estimation and hyperparameter selection are restricted to information available before the held-out test period; rolling-origin graph screens use only prior history.

### Figure 2. Held-out RMSE comparison
**Caption:** Test-set RMSE for graph message passing, no-graph and random-graph architectures. The near-identical graph and random-graph performance motivates the topology-specific statistical comparison.

### Figure 3. Training-only graph screen
**Caption:** Directed predictive-dependence structure estimated without access to held-out test observations. Edges represent Granger predictive dependence rather than structural causal effects.

### Figure 4. Forecast comparison
**Caption:** Actual and predicted next-day S&P 500 log returns for the held-out test period. Predictions are generated without using future test observations for graph construction or hyperparameter selection.

### Figure 5. HAC sensitivity
**Caption:** Forecast-comparison inference across HAC bandwidths 0, 5, 10 and 20. The primary bandwidth is 10; Holm correction is applied within the predeclared comparison family at each bandwidth.
