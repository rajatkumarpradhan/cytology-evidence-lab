# Cytology Evidence Lab

[![CI](https://github.com/rajatkumarpradhan/cytology-evidence-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rajatkumarpradhan/cytology-evidence-lab/actions/workflows/ci.yml)

> **Offline research demo** · Public historical data · Not a medical device or patient-care tool. Its Q&A assistant is deterministic, not an LLM.

**Start here:** [Run the project](#run) · [Evaluation design](#architecture-and-evaluation) · [Responsible-use limits](#limitations-and-responsible-use) · [Model card](MODEL_CARD.md)


A reproducible, offline data-science research project on the public [Wisconsin Diagnostic Breast Cancer dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). The aim is to evaluate a tabular classification workflow and explain its evidence, **not diagnose anyone**. The dataset has 569 historical records and 30 numerical measurements from digitized fine-needle aspirate images. The source is UCI (Wolberg, W., Mangasarian, O., Street, N., & Street, W.; see UCI citation and DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B)); Check the dataset page for UCI's current license terms before reuse. It is loaded from scikit-learn's packaged copy, so no data download, cloud credentials, API key or GPU is needed. Malignant is explicitly mapped to the positive class.

## Run

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[test]'
pytest -q
cytology-lab train --output artifacts
cytology-lab ask "How was the model selected?" --report artifacts/report.json
cytology-lab ask "What are the limitations?" --report artifacts/report.json
```

The training command writes `artifacts/report.json`, `artifacts/model.joblib`, and `artifacts/test_probabilities.png`. Output artifacts are generated, not checked in. `--quick` reduces CV folds and permutation repeats for a smoke run and must not be described as the final benchmark. Re-running with the same package versions and seed should reproduce metrics, subject to platform numeric-library differences. The full run generally takes a few minutes on a laptop.

## Architecture and evaluation

```
Packaged UCI WDBC -> stratified 80/20 holdout (seed 42)
                    -> training-only EDA and row-wise ratio features
                    -> nested stratified CV: inner grid tuning, outer family comparison
                       (scaled logistic regression vs random forest)
                    -> refit best hyperparameters on training only
                    -> training-fold sigmoid probability calibration
                    -> cross-fitted training probabilities -> F2 threshold
                    -> single untouched holdout assessment
                       ROC AUC, PR average precision, Brier, sensitivity,
                       specificity, confusion counts, bootstrap ROC AUC interval
                    -> permutation feature importance on original holdout columns
                    -> JSON evidence -> local question router with field citations
```

The `RatioFeatures` transformer adds worst/mean radius, worst/mean area, and worst/mean concave-point ratios. All scaling is inside the estimator pipeline and fit within training folds. The holdout never chooses hyperparameters or the threshold. The per-family outer CV scores are model-selection estimates and are **not** an independent unbiased final estimate of the overall selection procedure; the untouched test is the final reported estimate. The ROC AUC bootstrap interval resamples the held-out predictions and does not capture retraining variance. Permutation importance is descriptive, test-only, and may divide importance across correlated measurements; it is not causal or a patient-level explanation. The histogram is a diagnostic visualization, not a calibration plot.

The local assistant is deterministic, not a language model or a RAG system. It routes questions to recorded JSON fields, cites their paths, refuses patient/diagnosis requests, and abstains on unsupported questions. There is no paid API and no hidden external service. `model.joblib` is a local Python pickle artifact; load only artifacts you made yourself, never untrusted joblib files.

## Limitations and responsible use

This is a small, historical and selected cohort, with no prospective, demographic subgroup, temporal, institution-external, or clinical validation. It is not a medical device and cannot support patient care. High held-out accuracy on this dataset would not establish safety, fairness, or real-world performance. No patient records are collected or transmitted. For a production research extension, add independent external cohorts, shift checks, pre-registered thresholds, clinician review, and governance before any clinical discussion.

## Repository map

- `cytology_lab.py`: data, EDA, models, nested validation, calibration, metrics, plot, and grounded assistant
- `test_cytology_lab.py`: schema, data semantics, row-local features, estimators, assistant safety and end-to-end smoke test
- `pyproject.toml`: package and dependencies
- `.gitignore`: excludes generated models and caches

No license is asserted for this project's source code; UCI's dataset license is separate.
