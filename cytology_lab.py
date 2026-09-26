"""Reproducible WDBC research. Never use these predictions for medical decisions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, fbeta_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
DATASET = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
FEATURES = list(load_breast_cancer().feature_names)


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Append three scale-invariant, domain-inspired ratios with fixed semantics.

    Only reads an individual row: no corpus statistics or label information.
    """
    pairs = ((20, 0), (23, 3), (27, 7))  # worst/mean radius, area, concave points

    def fit(self, X, y=None):
        if np.asarray(X).shape[1] != 30:
            raise ValueError("Expected 30 original WDBC features")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 30 or not np.isfinite(X).all():
            raise ValueError("Expected finite (n, 30) feature matrix")
        ratios = [X[:, a] / np.maximum(np.abs(X[:, b]), 1e-8)
                  for a, b in self.pairs]
        return np.column_stack((X, *ratios))

    def get_feature_names_out(self, input_features=None):
        original = np.asarray(input_features if input_features is not None else FEATURES)
        return np.concatenate((original, ["worst/mean radius", "worst/mean area",
                                          "worst/mean concave points"]))


def data():
    ds = load_breast_cancer()
    # sklearn labels malignant=0, benign=1; invert so positive class is malignant.
    return ds.data, (ds.target == 0).astype(int)


def candidates(seed=SEED):
    return {
        "logistic": (Pipeline([("ratios", RatioFeatures()),
                               ("scale", StandardScaler()),
                               ("model", LogisticRegression(max_iter=3000,
                                                            random_state=seed))]),
                     {"model__C": [0.03, 0.3, 3.0],
                      "model__class_weight": [None, "balanced"]}),
        "forest": (Pipeline([("ratios", RatioFeatures()),
                             ("model", RandomForestClassifier(n_estimators=80,
                                                               n_jobs=1,
                                                               random_state=seed))]),
                   {"model__max_depth": [4, None],
                    "model__min_samples_leaf": [2, 5]})
    }


def safe_auc(y, p):
    return float(roc_auc_score(y, p))


def bootstrap_auc(y, p, seed=SEED, repeats=400):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            vals.append(safe_auc(y[idx], p[idx]))
    return [float(v) for v in np.quantile(vals, [0.025, 0.975])]


def choose_threshold(y, p):
    # Optimizes sensitivity-weighted F2 on training-only out-of-fold probabilities.
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y, p >= t, beta=2, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def nested_selection(X, y, quick=False):
    """Outer validation estimates model-family selection without holdout access."""
    outer = StratifiedKFold(n_splits=2 if quick else 3, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=2 if quick else 3, shuffle=True, random_state=SEED+1)
    results = {}
    for name, (pipe, grid) in candidates().items():
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1)
        scores = cross_validate(search, X, y, scoring={"roc_auc": "roc_auc",
                                                       "average_precision": "average_precision"},
                                cv=outer, n_jobs=1)
        results[name] = {"outer_roc_auc": [float(v) for v in scores["test_roc_auc"]],
                         "outer_average_precision": [float(v) for v in scores["test_average_precision"]],
                         "mean_outer_roc_auc": float(np.mean(scores["test_roc_auc"]))}
    # Fixed deterministic tie-break: logistic first.
    selected = max(results, key=lambda k: results[k]["mean_outer_roc_auc"])
    return selected, results, inner


def eda(X, y):
    corr = np.corrcoef(X, rowvar=False)
    upper = np.triu(np.ones(corr.shape, dtype=bool), 1)
    pairs = np.argwhere(upper)
    strongest = sorted(((float(corr[i, j]), FEATURES[i], FEATURES[j]) for i, j in pairs),
                       key=lambda item: abs(item[0]), reverse=True)[:5]
    return {"rows": int(len(y)), "features": int(X.shape[1]),
            "malignant": int(y.sum()), "benign": int(len(y)-y.sum()),
            "missing_or_nonfinite": int((~np.isfinite(X)).sum()),
            "top_correlations": [{"r": r, "a": a, "b": b} for r, a, b in strongest]}


def train(output: Path, quick=False):
    output.mkdir(parents=True, exist_ok=True)
    X, y = data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2,
                                          stratify=y, random_state=SEED)
    selected, nested, inner = nested_selection(Xtr, ytr, quick)
    pipe, grid = candidates()[selected]
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1)
    search.fit(Xtr, ytr)
    # Calibrator and threshold are trained on the training partition only.
    calibration_cv = StratifiedKFold(n_splits=2 if quick else 3, shuffle=True,
                                    random_state=SEED+2)
    proto = CalibratedClassifierCV(clone(search.best_estimator_),
                                   method="sigmoid", cv=calibration_cv)
    oof = cross_val_predict(proto, Xtr, ytr, cv=calibration_cv,
                            method="predict_proba")[:, 1]
    threshold = choose_threshold(ytr, oof)
    model = clone(proto).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    perm = permutation_importance(model, Xte, yte, scoring="roc_auc",
                                  n_repeats=5 if quick else 10,
                                  random_state=SEED, n_jobs=1)
    ranking = np.argsort(perm.importances_mean)[::-1][:10]
    report = {
        "dataset": DATASET, "seed": SEED, "task": "retrospective WDBC classification research",
        "positive_class": "malignant", "eda": eda(Xtr, ytr),
        "split": {"train": int(len(ytr)), "untouched_test": int(len(yte)),
                  "stratified": True},
        "selection": {"chosen": selected, "nested_cv_training_only": nested,
                      "best_parameters": search.best_params_, "best_inner_roc_auc": float(search.best_score_)},
        "calibration": {"method": "sigmoid, training-only CV",
                        "training_oof_brier": float(brier_score_loss(ytr, oof)),
                        "threshold": threshold,
                        "threshold_rule": "maximize F2 on training-only out-of-fold probabilities"},
        "test": {"roc_auc": safe_auc(yte, proba),
                 "roc_auc_bootstrap_95pct": bootstrap_auc(yte, proba),
                 "average_precision": float(average_precision_score(yte, proba)),
                 "brier": float(brier_score_loss(yte, proba)),
                 "sensitivity": float(tp/(tp+fn)),
                 "specificity": float(tn/(tn+fp)),
                 "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}},
        "explanation": {"method": "test-set permutation importance (ROC AUC decrease)",
                        "not_causal": True,
                        "top_original_features": [
                            {"feature": FEATURES[i], "mean_auc_drop": float(perm.importances_mean[i]),
                             "std": float(perm.importances_std[i])} for i in ranking]},
        "limitations": ["Small historical dataset, no external validation",
                        "Correlated features can dilute permutation importance",
                        "Not a diagnostic tool or clinical decision support"]}
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURES},
                output/"model.joblib")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(proba[yte == 0], bins=np.linspace(0, 1, 21), alpha=.6, label="Benign")
    ax.hist(proba[yte == 1], bins=np.linspace(0, 1, 21), alpha=.6, label="Malignant")
    ax.axvline(threshold, color="black", linestyle="--", label="Training-selected threshold")
    ax.set(xlabel="Estimated malignant-class probability", ylabel="Held-out cases",
           title="Held-out probability distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"test_probabilities.png", dpi=150)
    plt.close(fig)
    return report


def answer(question: str, report: dict) -> str:
    """Local evidence assistant: routes questions to report fields, cites JSON paths.

    No LLM, RAG, medical advice, or unverifiable free-form metric generation.
    """
    q = question.lower()
    if re.search(r"diagnos|patient|treat|medical advice|should i|my result", q):
        return "I cannot interpret a patient result or offer medical advice. This is retrospective research only."
    if any(t in q for t in ("source", "dataset", "data")):
        return (f"Dataset: {report['dataset']} [report.json:dataset]. Training rows: "
                f"{report['split']['train']}; held-out rows: {report['split']['untouched_test']} "
                "[report.json:split].")
    if any(t in q for t in ("model", "select", "validation", "tuning")):
        s = report["selection"]
        return (f"Chosen family: {s['chosen']} [report.json:selection.chosen]. "
                "Nested cross-validation used the training partition only "
                "[report.json:selection.nested_cv_training_only].")
    if any(t in q for t in ("threshold", "calibrat")):
        c = report["calibration"]
        return (f"Threshold {c['threshold']:.2f}, selected to maximize F2 using "
                "training-only out-of-fold probabilities [report.json:calibration].")
    if any(t in q for t in ("importance", "explain", "feature")):
        e = report["explanation"]
        names = ", ".join(v["feature"] for v in e["top_original_features"][:3])
        return (f"Leading held-out permutation features: {names} "
                "[report.json:explanation.top_original_features]. This is not causal evidence.")
    if any(t in q for t in ("performance", "score", "auc", "sensitivity", "error")):
        t = report["test"]
        return (f"Held-out ROC AUC {t['roc_auc']:.3f}, average precision "
                f"{t['average_precision']:.3f}, sensitivity {t['sensitivity']:.3f}; "
                f"false negatives {t['confusion']['fn']} [report.json:test]. "
                "These are small-sample research estimates, not clinical validation.")
    if any(t in q for t in ("limit", "safe", "deploy")):
        return "Limitations: " + "; ".join(report["limitations"]) + " [report.json:limitations]."
    return ("I can answer questions about dataset, model selection, test performance, "
            "threshold, feature importance, and limitations. I cannot infer other facts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="run reproducible pipeline")
    t.add_argument("--output", type=Path, default=Path("artifacts"))
    t.add_argument("--quick", action="store_true", help="faster CV smoke run, not final evaluation")
    a = sub.add_parser("ask", help="query a saved, cited report locally")
    a.add_argument("question")
    a.add_argument("--report", type=Path, default=Path("artifacts/report.json"))
    args = parser.parse_args()
    if args.command == "train":
        result = train(args.output, args.quick)
        print(json.dumps({"model": result["selection"]["chosen"],
                          "test": result["test"], "output": str(args.output)}, indent=2))
    else:
        print(answer(args.question, json.loads(args.report.read_text())))


if __name__ == "__main__":
    main()
