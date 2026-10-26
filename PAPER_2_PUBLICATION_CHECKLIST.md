# Paper 2 — Publication Preparation Checklist

## Claims audit

- [x] State the Granger graph as predictive dependence, not structural causation.
- [x] State that graph construction is leakage-controlled.
- [x] Distinguish numerical RMSE improvement from statistically supported improvement.
- [x] Include topology-matched random-graph ablation.
- [x] Report Holm-adjusted p-values for the four predeclared comparisons.
- [x] Avoid claiming universal failure of graph/causal forecasting.
- [x] Treat rolling-origin and regime analyses as robustness evidence rather than independent confirmatory tests.
- [x] Keep the untouched test period separate from model selection.

## Numerical consistency

Primary reported test metrics:

- Graph message passing: RMSE 0.01119991; MAE 0.00744418; R² -0.04871.
- No-graph control: RMSE 0.01131990; MAE 0.00741144; R² -0.07130.
- Random graph: RMSE 0.01120288; MAE 0.00740995; R² -0.04926.
- Test observations: 3,807.
- Final graph edges: 122 across 14 variables.

Primary paired HAC comparisons:

- Graph vs no graph: raw p 0.5316; Holm p 1.0000.
- Graph vs random graph: raw p 0.9732; Holm p 1.0000.
- Graph vs training mean: raw p 0.1134; Holm p 0.4538.
- Graph vs zero return: raw p 0.1231; Holm p 0.4538.

## Presentation requirements

1. Abstract must lead with the research question and rigorous evaluation design.
2. Introduction must identify leakage, capacity and topology confounding as threats to graph claims.
3. Methods must make chronological information flow explicit.
4. Results must present the random-graph comparison immediately beside the no-graph comparison.
5. Statistical inference must be described as paired HAC loss comparison with Holm correction.
6. Discussion must explain why numerical improvement is insufficient evidence of graph-specific value.
7. Limitations must state that results are conditional on dataset, variables, horizon, graph estimator and architecture.
8. Conclusion must remain a null finding for graph-specific improvement.

## Figure plan

- Figure 1: chronological information-flow/evaluation protocol.
- Figure 2: test RMSE and MAE for graph, no-graph and random-graph models.
- Figure 3: training-only directed graph.
- Figure 4: test-period forecasts and residuals.
- Figure 5: HAC sensitivity across bandwidths 0, 5, 10 and 20.

## Final pre-submission checks still required

- Verify all citations and bibliography entries against primary/authoritative sources.
- Verify exact data provenance and licensing statement.
- Verify that every figure is generated from committed result files.
- Run the final CI workflow from a clean checkout.
- Check manuscript against the target venue's word count, formatting and reference style.
- Perform a final line-by-line numerical claims audit against the saved results bundle.
