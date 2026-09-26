"""Saved-report schema and seed-determinism checks.

Policy: these tests validate structure, provenance fields and numeric ranges -
never a specific flattering metric value. A metric pinned to a constant would
hide regressions and reward overfitting to the test.
"""
import json
import math

from cytology_lab import SEED, train

METRIC_KEYS = ("roc_auc", "average_precision", "brier", "sensitivity", "specificity")


def _quick_reports(tmp_path):
    first = train(tmp_path / "run1", quick=True)
    second = train(tmp_path / "run2", quick=True)
    stored = json.loads((tmp_path / "run1" / "report.json").read_text())
    return first, second, stored


def test_seed_makes_quick_runs_bit_identical(tmp_path):
    first, second, stored = _quick_reports(tmp_path)
    assert first == stored  # CLI persists exactly what it computed
    assert first == second  # same seed, same inputs -> identical evidence
    assert stored["seed"] == SEED == 42


def test_report_schema_and_ranges(tmp_path):
    _, _, report = _quick_reports(tmp_path)
    assert set(report) == {"dataset", "seed", "task", "positive_class", "eda",
                           "split", "selection", "calibration", "test",
                           "explanation", "limitations"}
    assert report["dataset"].startswith("https://archive.ics.uci.edu/")
    assert report["positive_class"] == "malignant"
    assert report["eda"]["missing_or_nonfinite"] == 0

    split = report["split"]
    assert split["stratified"] is True
    assert split["train"] == 455 and split["untouched_test"] == 114

    selection = report["selection"]
    assert selection["chosen"] in {"logistic", "forest"}
    assert set(selection["nested_cv_training_only"]) == {"logistic", "forest"}
    assert 0 < selection["best_inner_roc_auc"] <= 1

    calibration = report["calibration"]
    assert 0.05 <= calibration["threshold"] <= 0.95
    assert 0 <= calibration["training_oof_brier"] <= 1
    assert "training" in calibration["threshold_rule"]

    test = report["test"]
    for key in METRIC_KEYS:
        value = test[key]
        assert math.isfinite(value) and 0 <= value <= 1, key
    lo, hi = test["roc_auc_bootstrap_95pct"]
    assert 0 <= lo <= hi <= 1
    confusion = test["confusion"]
    assert set(confusion) == {"tn", "fp", "fn", "tp"}
    assert sum(confusion.values()) == split["untouched_test"]
    assert confusion["tp"] + confusion["fn"] > 0  # denominators are real
    # Guard against nonsense without pinning a flattering constant:
    # any non-broken model must beat chance on this dataset.
    assert test["roc_auc"] > 0.5
    prevalence = (confusion["tp"] + confusion["fn"]) / split["untouched_test"]
    assert test["average_precision"] >= prevalence  # beats a random ranking
    explanation = report["explanation"]
    assert explanation["not_causal"] is True
    assert len(explanation["top_original_features"]) == 10
    assert report["limitations"]  # honest limits are part of the schema


def test_artifacts_written(tmp_path):
    train(tmp_path, quick=True)
    assert (tmp_path / "report.json").stat().st_size > 2000
    assert (tmp_path / "model.joblib").stat().st_size > 1000
    assert (tmp_path / "test_probabilities.png").stat().st_size > 1000
