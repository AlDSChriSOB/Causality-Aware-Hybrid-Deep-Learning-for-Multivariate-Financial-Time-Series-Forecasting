# Cross-Paper Evidence Audit — Version 1

**Audit date:** 2026-10-25  
**Scope:** Repository metadata, README claims, available file contents and prior notebook inspection notes.  
**Important:** This is an evidence triage, not a completed rerun. A claim in a README is not treated as independently reproduced evidence.

## Executive finding

The four projects are promising starting points, but the current repositories do not yet support four submission-ready manuscripts. The most consequential issues are (i) Git LFS pointer files prevent access to the Paper 1 raw data and notebooks through the current repository content interface, (ii) Paper 2 has previously identified in-sample evaluation and invalid/unsuitable metrics, (iii) Paper 3 reports conflicting COVID shock/recovery dates, and (iv) Paper 4's headline results are documented in the README but have not been independently regenerated from the workbook in this audit.

## Evidence ledger

| Paper | Existing evidence | Current trust status | Required action |
|---|---|---|---|
| 1. Social-media sentiment | README reports 1.6 million tweets, 98% accuracy, peak positivity at 1 AM, peak negativity at 4 PM, and six clusters. Repository CSV and two notebook files inspected through GitHub contents API are Git LFS pointer stubs, not the underlying bytes. | **Claims not independently reproducible from currently accessible files.** The README contains internally inconsistent framing: Sentiment140's original labels are binary, while the README also describes VADER relabeling and a hybrid multi-class temporal task. The claimed 98% must be withheld until target definition and held-out evaluation are verified. | Restore/obtain actual LFS objects; inspect full dissertation and notebooks; reconstruct labels; run leakage-safe baselines and temporal tests. |
| 2. S&P 500 forecasting | Notebook and SP500.csv exist in repository. Prior notebook inspection found predictions evaluated on training inputs, MAPE used for returns, implausibly high hybrid directional accuracy, and a t-test on squared-error differences presented as a Diebold–Mariano test. Graph may be conceptual rather than connected to the neural model. | **Existing reported neural results are not trustworthy as out-of-sample evidence.** Data access through the current file connector did not return CSV bytes in this audit, so no corrected metrics are reported here. | Build a chronological experiment pipeline; check target alignment; compare baselines; fit transforms on training only; implement correct loss-differential inference; rerun on actual CSV. |
| 3. Transborder mobility | README says COVID onset April 2020, recovery September 2020 and TTN 5 months. Prior notebook inspection notes a cell reporting onset 2020-02 and recovery 2020-03 with TTN 1 month. README says BTS data 1996–2024. | **COVID TTN and event dating are contradictory and must not be published until reconciled.** Claims comparing post-2008 and post-COVID recovery are unverified absent comparable definitions and uncertainty. | Retrieve and run notebook with official data; specify exact series, onset rule, recovery threshold and persistence rule; compare detectors and thresholds; quantify uncertainty. |
| 4. UK house prices | README documents HM Land Registry/ONS data from Jan 1995–Nov 2025, nominal prices, 371 months, and several headline borough/regional values and charts. Workbook is present in repository, but its contents and notebook outputs were not independently executed in this audit. | **Descriptive README values are provisional, not independently verified.** Nominal growth does not establish real appreciation, convergence/divergence, or causal shock effects. | Load workbook in a reproducible environment; validate sheets, dates, units and aggregates; regenerate all tables/figures; add inflation-adjusted analysis and formal dispersion/break tests. |

## Results that should not currently appear as established findings

1. Paper 1: 98% accuracy, 0.99 AUC, and specific peak-hour/country claims until regenerated from a documented dataset, target, split, and statistical procedure.
2. Paper 2: any assertion that the hybrid model beats baselines out of sample; any 99.995% directional-accuracy claim; any p-value from the existing one-sample t-test described as a Diebold–Mariano test.
3. Paper 3: the five-month COVID recovery claim, one-month recovery claim, and crisis-to-crisis recovery ranking until the code and definitions agree.
4. Paper 4: specific borough price/CAGR figures and regional divergence conclusions until the workbook-derived outputs are reproduced and definitions confirmed.

## Reproducibility blockers and resolution

### Paper 1 — Git LFS
The GitHub contents API returns LFS pointer text for `sentiment140_Dataset.csv` (declared LFS object size 238,803,811 bytes) and two notebook files (one declared 6,601,936 bytes). The actual object bytes were not available from the returned contents. The raw data must be downloaded from the authorised LFS source or supplied as a mounted file before a genuine rerun can be completed.

### Paper 2 — experiment execution
The current file interface did not return the SP500.csv contents during this audit. A corrected script can be prepared and committed, but the final numerical result table must be generated from the real data and run logs. No simulated or guessed results should be substituted.

### Paper 3 — source data
The repository's `data_setup.py` / data retrieval path must be verified and the current official dataset snapshot recorded. Because government data may be revised, record download date, source endpoint, query/filter, checksum, and selected series.

### Paper 4 — Excel source
The workbook must be loaded and checked in Python with `openpyxl`/`pandas`; all outputs should be generated from the workbook rather than copied from README prose.

## Publication readiness by paper

| Requirement | Paper 1 | Paper 2 | Paper 3 | Paper 4 |
|---|---|---|---|---|
| Data bytes verified | Blocked by LFS pointer | Not yet available to current runner | Source retrieval required | Workbook not executed |
| Main results reproduced | No | No; existing evaluation has critical flaws | No; contradictory dates | No |
| Corrected statistical analysis | Pending | Pending | Pending | Pending |
| Manuscript results section can be finalised | No | No | No | No |
| Submission-ready now | No | No | No | No |

## Integrity policy

Do not create numerical result tables from README claims alone. Do not invent confidence intervals, p-values, dataset counts, journal acceptance likelihood, citations, or author declarations. Drafts may be developed in parallel, but the final abstract and results must be locked only after corrected code produces reproducible outputs.

## Immediate next steps

1. Make the underlying datasets and code accessible to the execution environment (resolve Paper 1 LFS, retrieve Paper 2 CSV, retrieve official BTS snapshot, load Paper 4 workbook).
2. Add clean experiment scripts and automated tests to each repository.
3. Run experiments, store machine-generated CSV tables/figures and environment metadata, and record commit hashes.
4. Draft each manuscript from the locked results and verify each reference against its publisher/DOI.
5. Select target venues only after the true contribution and final evidence are clear.