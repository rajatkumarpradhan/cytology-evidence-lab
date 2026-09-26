# Model Card: WDBC Retrospective Classifier

## Overview

A reproducible research pipeline that trains and evaluates a tabular classifier
on the public Wisconsin Diagnostic Breast Cancer (WDBC) dataset, with a
deterministic local Q&A assistant that answers questions about the saved
evidence by citing `report.json` fields. This is an engineering study of
evaluation hygiene, not a medical product.

## Intended use and out-of-scope use

- **Intended:** demonstrating reproducible ML evaluation (nested CV,
  calibration, threshold policy, cited evidence) on public historical data.
- **Out of scope:** patient care, diagnosis, screening, treatment decisions,
  or any clinical deployment. There is **no prospective validation and no
  validation on data from any other institution**. The assistant refuses
  patient-interpretation questions by design.

## Dataset provenance

- Source: UCI Machine Learning Repository, "Breast Cancer Wisconsin
  (Diagnostic)", DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B),
  Wolberg, Mangasarian, Street & Street; licensed CC BY 4.0 per UCI.
- 569 rows, 30 numeric features from digitized fine-needle aspirate images;
  212 malignant / 357 benign.
- Loaded from scikit-learn's packaged copy (`load_breast_cancer`); no
  downloads, credentials, or patient-identifiable data are involved.

## Positive-class mapping

scikit-learn encodes `target` 0 = malignant, 1 = benign. The pipeline inverts
this so **the positive class is malignant** (`y = (target == 0)`), which is the
clinically relevant event for sensitivity reporting. A unit test pins the
expected 212 malignant count so the mapping cannot silently flip.

## Split and validation design

- One stratified 80/20 holdout (seed 42): 455 training rows, 114 untouched
  test rows. All tuning, selection, calibration and threshold fitting happen on
  the training partition only.
- Model-family selection uses nested stratified CV inside the training set
  (outer loop compares scaled logistic regression vs random forest; inner loop
  tunes hyperparameters).
- Permutation importance is computed on the untouched holdout's original 30
  columns. It is associational, **not causal**.

## Threshold selection

The operating threshold maximizes F2 (sensitivity-weighted) over
training-only, cross-fitted out-of-fold probabilities. The test set never
influences the threshold. The exact value is recorded in
`report.json:calibration.threshold` per run rather than hard-coded anywhere.

## Calibration

Probabilities are calibrated with a sigmoid (Platt) calibrator cross-fitted on
training folds only. Calibration quality is reported as Brier score on both the
training out-of-fold probabilities and the untouched holdout. Small samples
limit how well calibration can be assessed; treat the holdout Brier as
indicative, not definitive.

## Metrics reported

ROC AUC (with a 95% bootstrap interval), PR average precision, Brier score,
sensitivity, specificity, and confusion counts on the single untouched
holdout, plus nested-CV selection scores. All values are recomputed per run;
tests check schema, ranges and determinism, never a pinned target number.

## Limitations

- Small historical cohort from a single source; no external or prospective
  validation.
- Correlated features can dilute and shift permutation importance.
- A fixed seed gives reproducibility, not robustness across data shifts.
- The `--quick` mode is a smoke configuration, not a benchmark.
- Not a diagnostic tool or clinical decision support.

## Reproduce

```bash
python -m pip install -e '.[test]'
pytest -q
cytology-lab train --output artifacts
```
