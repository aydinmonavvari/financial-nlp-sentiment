"""Evaluation utilities: metrics, McNemar test, error-analysis helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import LABELS


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(accuracy_score(list(y_true), list(y_pred)))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(list(y_true), list(y_pred), average="macro"))


def per_class_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Per-class precision/recall/F1/support, indexed by canonical label order."""
    labels = list(LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        list(y_true), list(y_pred), labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
        index=labels,
    )


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> np.ndarray:
    """Confusion matrix with rows = true, cols = predicted, canonical order."""
    return confusion_matrix(list(y_true), list(y_pred), labels=list(LABELS))


def multiclass_brier(y_true: Sequence[str], proba: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2.

    Range [0, 2] for 3 classes; lower is better. Used for the calibration
    note - a ranking model can be accurate while poorly calibrated, and this
    scalar is the honest check on absolute probability quality.
    """
    proba = np.asarray(proba, dtype=float)
    idx = np.array([list(LABELS).index(y) for y in y_true])
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(idx)), idx] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def mean_confidence(proba: np.ndarray) -> float:
    """Mean maximum-class probability - the crudest calibration sanity check."""
    return float(np.max(np.asarray(proba, dtype=float), axis=1).mean())


def mcnemar_test(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
) -> dict:
    """Exact McNemar test on paired correctness of two systems.

    Builds the 2x2 correctness table (A correct/wrong x B correct/wrong) and
    runs the exact binomial test on the off-diagonal counts b (A right, B
    wrong) and c (A wrong, B right) via
    ``statsmodels.stats.contingency_tables.mcnemar``. Two-sided p-value.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    y_true = list(y_true)
    a_ok = np.array([p == t for p, t in zip(pred_a, y_true, strict=True)])
    b_ok = np.array([p == t for p, t in zip(pred_b, y_true, strict=True)])
    n00 = int(np.sum(~a_ok & ~b_ok))
    n01 = int(np.sum(~a_ok & b_ok))
    n10 = int(np.sum(a_ok & ~b_ok))
    n11 = int(np.sum(a_ok & b_ok))
    table = np.array([[n11, n10], [n01, n00]])  # rows: B outcome, cols: A outcome
    result = mcnemar(table, exact=True)
    return {
        "b_a_right_b_wrong": n10,
        "c_a_wrong_b_right": n01,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def collect_misclassified(
    texts: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    max_examples: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample misclassified test sentences for the error-analysis section.

    Rounds across confusion pairs ordered by frequency (worst-confused pairs
    first), taking one deterministic example per pair per round, until
    ``max_examples`` rows are collected.
    """
    rng = np.random.default_rng(seed)
    rows = [
        {"sentence": str(t), "true": tt, "predicted": pp}
        for t, tt, pp in zip(texts, y_true, y_pred, strict=True)
        if tt != pp
    ]
    if not rows:
        return pd.DataFrame(columns=["sentence", "true", "predicted"])
    frame = pd.DataFrame(rows)
    frame["pair"] = frame["true"] + "->" + frame["predicted"]
    pair_order = frame["pair"].value_counts().index.tolist()

    picked: list[int] = []
    for _round in range(max_examples):
        for pair in pair_order:
            if len(picked) >= max_examples:
                break
            remaining = frame.index[
                (frame["pair"] == pair) & (~frame.index.isin(picked))
            ].tolist()
            if remaining:
                picked.append(int(remaining[int(rng.integers(len(remaining)))]))
    out = frame.loc[picked[:max_examples]].drop(columns=["pair"]).reset_index(drop=True)
    return out


def classification_report_text(y_true: Sequence[str], y_pred: Sequence[str]) -> str:
    return classification_report(
        list(y_true), list(y_pred), labels=list(LABELS), zero_division=0
    )


__all__ = [
    "accuracy",
    "classification_report_text",
    "collect_misclassified",
    "confusion",
    "macro_f1",
    "mean_confidence",
    "mcnemar_test",
    "multiclass_brier",
    "per_class_metrics",
]
