# Four-Paper Research Programme: Execution Plan

**Owner:** Christian Somtoo Obiechina  
**Purpose:** Convert four existing projects into reproducible, defensible research manuscripts.  
**Status:** Research audit and experiment redesign in progress. No new experiment result should be described as established until it has been run and recorded.

## Working rules
1. Preserve the original notebooks and results; create a separate reproducible pipeline for corrected experiments.
2. Do not fabricate results, citations, data provenance, or claims of statistical significance.
3. Use chronological splits for temporal data, fit transformations only on training data, and record random seeds and package versions.
4. Report uncertainty and appropriate baselines; avoid selecting a model or threshold using the final test set.
5. Treat observational associations as associations, not causal effects, unless identification assumptions are justified.
6. Keep a result ledger with dataset version, sample dates, code commit, split, metric, confidence interval, and notes for every reported number.
7. A manuscript is submission-ready only after its tables and claims match the final reproducible outputs and the target journal's formatting and ethics requirements.

## Paper 1 — Temporal dynamics of social-media sentiment
**Repository:** AlDSChriSOB/msc-research-project01  
**Provisional title:** *Temporal Dynamics of Sentiment in Social Media: A Reproducible Analysis of Diurnal Patterns and Text Classification*

### Audit questions
- Distinguish original Sentiment140 labels from any labels created by VADER. Document the exact target used in every experiment.
- Verify data provenance, sample size, date coverage, class distribution, duplicates, missing text, and preprocessing.
- Rebuild splits before resampling; ensure SMOTE or any resampling is applied only inside training folds and only to suitable vectorised representations.
- Inspect the claimed attention layer to verify that its output contributes to the prediction path.
- Separate classification performance from temporal-pattern analysis. A high random-split score alone does not validate generalisation or diurnal claims.
- Confirm timezone handling, timestamp coverage, and whether hour-of-day effects can be estimated from the data as supplied.

### Required experiments
- Transparent majority-class and lexical baselines; TF-IDF logistic regression / linear SVM; neural baseline; proposed hybrid only if its implementation is valid.
- Stratified random split for comparability plus a temporal or source-aware holdout where timestamps and labels support it.
- Per-class precision, recall, F1, macro-F1, confusion matrix, calibration where appropriate, and confidence intervals.
- Ablations for architecture components and embeddings; leakage checks and seed sensitivity.
- Diurnal analysis with counts, uncertainty intervals, weekday/hour controls where justified, multiple-comparison correction, and sensitivity to filtering.

### Manuscript contribution
The primary contribution should be a careful, reproducible analysis of sentiment variation over time, with text classification serving as a validated measurement component. Do not frame an unverified headline accuracy as the principal scientific contribution.

## Paper 2 — Causality-informed financial forecasting
**Repository:** this repository.  
**Provisional title:** *Causality-Informed Hybrid Deep Learning for S&P 500 Return Forecasting Under Market Regime Changes*  
**Design document:** [PAPER_2_RESEARCH_DESIGN.md](PAPER_2_RESEARCH_DESIGN.md)

### Critical corrections
- Replace in-sample predictions with strictly chronological train/validation/test or rolling-origin evaluation.
- Fit scalers and feature selection on training windows only.
- Predict returns, not nonstationary price levels, as the primary target; define forecast horizon and information available at forecast time.
- Do not call a hand-drawn graph or contemporaneous association a causal mask. Implement and test a graph-constrained model if that is the claimed novelty.
- Correct the directional-accuracy calculation and inspect target/prediction alignment and prediction distributions.
- Do not use MAPE as a primary metric for returns near zero.
- Replace the one-sample t-test on squared-error differences with a properly specified Diebold–Mariano procedure, including a suitable long-run variance estimator where forecast errors overlap or are serially dependent.
- Control for multiple testing in lag discovery and estimate the graph on training data only.

### Required experiments
Naive zero-return and historical-mean forecasts; regularised linear model; AR benchmark where appropriate; tree-based model; LSTM; attention LSTM; graph-constrained model. Compare MAE/RMSE, out-of-sample R-squared, directional accuracy, uncertainty intervals, regime-specific results, and ablations. Use rolling-origin evaluation and preserve a final untouched test period.

## Paper 3 — Transborder mobility and systemic shocks
**Repository:** AlDSChriSOB/Unsupervised-Machine-Learning-Approach-to-Detecting-Systemic-Shocks-in-Transborder-Mobility  
**Provisional title:** *Detecting Structural Shocks and Recovery in U.S. Border Mobility: Robustness of Unsupervised Time-Series Methods*

### Critical corrections
- Reconcile the contradictory reported COVID onset/recovery dates and the one-month versus five-month time-to-normalisation claims before drafting the results.
- Document the exact BTS series, unit, geography, frequency, extraction date, revisions, missingness, and why the chosen passenger series is an appropriate mobility indicator.
- Distinguish anomaly detection from causal shock identification.
- Specify STL period, trend/seasonal windows, Isolation Forest contamination, fitting period, threshold, and recovery rule before comparing crises.
- Do not assert slower recovery after one crisis than another without a predeclared, comparable definition and uncertainty analysis.

### Required experiments
Compare robust seasonal baselines, STL residual thresholds, Isolation Forest, and change-point methods where feasible. Vary decomposition settings and anomaly thresholds; compare alternative border-crossing series; bootstrap or otherwise quantify uncertainty in onset, nadir, and recovery dates. Test whether conclusions persist under reasonable recovery definitions.

## Paper 4 — UK house-price divergence
**Repository:** AlDSChriSOB/uk-house-price-index-analysis  
**Provisional title:** *Regional Divergence in UK House Prices: Long-Run Growth, Structural Breaks, and Market Resilience, 1995–2025*

### Audit questions
- Verify source, geography, date range, revision status, definitions, missingness, and whether values are nominal or inflation-adjusted.
- Identify the level of aggregation (nation, region, local authority, London borough) and avoid pooling unlike geographies without an explicit design.
- Separate price-index changes from average-price composition effects; describe which series is used for each question.
- Avoid interpreting descriptive growth charts as evidence of statistically significant divergence or convergence.

### Required experiments
- Reproduce descriptive trends and growth rates with transparent denominators.
- Compare nominal and inflation-adjusted prices if suitable CPI/RPI data are obtained and documented.
- Analyse regional dispersion over time and test divergence/convergence with methods appropriate to the panel and integration properties.
- Test structural breaks and sensitivity to start/end dates; report uncertainty and account for serial dependence.
- Where supported by the data, analyse heterogeneous recovery after major shocks without claiming causality.

## Shared deliverables
For each paper, produce:
1. AUDIT.md — data/code audit, known limitations, claim-to-evidence ledger.
2. A clean analysis pipeline or notebook that can be run from a documented environment.
3. results/ — generated tables and figures, not manually typed values.
4. MANUSCRIPT.md — abstract, introduction, related work, data, methods, results, discussion, limitations, conclusion, and references.
5. REPRODUCIBILITY.md — setup, data acquisition, run instructions, seeds, software versions, and output map.
6. Submission checklist — target journal fit, word/figure limits, references verified, declarations, and preprint/open-access options.

## Sequence and release gates
**Gate A — Audit:** inspect data, notebook cells, output provenance, and contradictions.  
**Gate B — Rebuild:** implement leakage-safe experiments and validated statistical methods.  
**Gate C — Results lock:** freeze analysis choices and generate final tables/figures from code.  
**Gate D — Manuscript:** write claims only from the locked results and cite verified literature.  
**Gate E — Submission preparation:** select a suitable venue, format, and complete required declarations.

Recommended sequence: Paper 1 (most mature and foundational), Paper 2 (highest technical correction burden), Paper 3 (resolve event/recovery definitions), Paper 4 (formalise regional divergence analysis). If the user's immediate priority is Paper 2, its design and audit can proceed in parallel.