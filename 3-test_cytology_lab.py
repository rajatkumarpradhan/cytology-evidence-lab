import json
import numpy as np
import pytest
from cytology_lab import (RatioFeatures, answer, bootstrap_auc, candidates, data,
                          eda, choose_threshold, train)


def test_dataset_semantics():
    X, y = data()
    assert X.shape == (569, 30)
    assert y.sum() == 212  # UCI malignant observations
    assert eda(X, y)["missing_or_nonfinite"] == 0


def test_ratios_rowwise_and_schema():
    X, _ = data()
    f = RatioFeatures().fit(X)
    transformed = f.transform(X[:2])
    assert transformed.shape == (2, 33)
    assert transformed[0, 30] == pytest.approx(X[0, 20] / X[0, 0])
    assert np.array_equal(transformed[0], f.transform(X[:1])[0])
    with pytest.raises(ValueError):
        f.transform(np.ones((1, 29)))


def test_pipelines_can_fit():
    X, y = data()
    for model, _ in candidates().values():
        model.fit(X[:120], y[:120])
        assert 0 <= model.predict_proba(X[121:122])[0, 1] <= 1


def test_threshold_and_bootstrap():
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .2, .8, .9])
    assert 0.05 <= choose_threshold(y, p) <= .95
    assert bootstrap_auc(y, p, repeats=50) == [1., 1.]


def test_grounded_assistant_never_gives_clinical_advice():
    report = {"dataset": "source", "split": {"train": 4, "untouched_test": 2},
              "selection": {"chosen": "forest"}, "calibration": {"threshold": .3},
              "test": {"roc_auc": .8, "average_precision": .7,
                       "sensitivity": .9, "confusion": {"fn": 1}},
              "explanation": {"top_original_features": [{"feature": "radius"}]},
              "limitations": ["Small dataset"]}
    assert "cannot interpret" in answer("diagnose my patient", report)
    assert "report.json:test" in answer("performance?", report)
    assert "cannot infer" in answer("who funded this?", report)


def test_end_to_end_quick(tmp_path):
    result = train(tmp_path, quick=True)
    stored = json.loads((tmp_path/"report.json").read_text())
    assert result == stored
    assert stored["split"] == {"train": 455, "untouched_test": 114,
                               "stratified": True}
    assert len(stored["selection"]["nested_cv_training_only"]) == 2
    assert (tmp_path/"model.joblib").stat().st_size > 1000
    assert (tmp_path/"test_probabilities.png").stat().st_size > 1000
