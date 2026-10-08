"""Offline tests for the financial sentiment benchmark.

Heavy dependencies (torch / transformers / sentence-transformers) are guarded
with ``pytest.importorskip`` so the suite passes on the light CI install
(base deps + dev extras). The two heavy smoke tests only run where the models
are available locally (they are cached after a study run).

The labeled sentences used here are SYNTHETIC FIXTURES written by hand for
this repository - they are test data, not research data.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from financial_nlp_sentiment.config import ID_TO_LABEL, LABEL_TO_ID, LABELS, StudyConfig
from financial_nlp_sentiment.data import (
    LABEL_COL,
    SENTENCE_COL,
    _parse_member,
    load_synthetic_fixture,
    make_splits,
)
from financial_nlp_sentiment.evaluation import (
    accuracy,
    collect_misclassified,
    confusion,
    macro_f1,
    mcnemar_test,
    multiclass_brier,
    per_class_metrics,
)
from financial_nlp_sentiment.models import (
    EmbeddingLogReg,
    TfidfLogReg,
    cv_macro_f1,
    majority_baseline,
    tune_c_on_validation,
)
from financial_nlp_sentiment.text import normalize_text, vader_predict, vader_to_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def fixture_frame() -> pd.DataFrame:
    return load_synthetic_fixture()


def toy_embed(texts: list[str]) -> np.ndarray:
    """Deterministic toy embedder: [pos-word count, neg-word count, length]."""
    pos_words = {"rose", "raised", "grew", "profit", "profitability", "strong", "increased"}
    neg_words = {"loss", "fell", "lowered", "decreased", "shrink", "sharply"}
    rows = []
    for t in texts:
        words = {w.strip(".,%0123456789").lower() for w in t.split()}
        rows.append([
            float(len(words & pos_words)),
            float(len(words & neg_words)),
            float(len(t)) / 100.0,
        ])
    return np.array(rows)


# ---------------------------------------------------------------------------
# preprocessing policy
# ---------------------------------------------------------------------------
def test_normalize_text_collapses_whitespace():
    assert normalize_text("  A\tB\n\nC  ") == "A B C"


def test_normalize_preserves_case():
    # Policy: NO lowercasing at the data layer. VADER is case-sensitive by
    # design; the TF-IDF vectorizer lowercases internally (documented).
    assert normalize_text("Operating PROFIT Rose") == "Operating PROFIT Rose"


# ---------------------------------------------------------------------------
# VADER lexicon baseline
# ---------------------------------------------------------------------------
def test_vader_to_label_thresholds_a_priori():
    config = StudyConfig()
    assert config.vader_pos_threshold == 0.35
    assert config.vader_neg_threshold == -0.15
    # boundary behavior: >= pos -> positive, <= neg -> negative
    assert vader_to_label(0.35) == "positive"
    assert vader_to_label(0.3499) == "neutral"
    assert vader_to_label(-0.15) == "negative"
    assert vader_to_label(-0.1499) == "neutral"
    assert vader_to_label(0.0) == "neutral"
    assert vader_to_label(1.0) == "positive"
    assert vader_to_label(-1.0) == "negative"


def test_vader_scores_simple_sentiment():
    _, pos = vader_predict(["Operating profit rose strongly and beat expectations."])
    _, neg = vader_predict(["The company reported a terrible loss and cut guidance."])
    assert pos[0] == "positive"
    assert neg[0] == "negative"


# ---------------------------------------------------------------------------
# labels and splits
# ---------------------------------------------------------------------------
def test_label_encoding_roundtrip():
    for lab in LABELS:
        assert ID_TO_LABEL[LABEL_TO_ID[lab]] == lab
    assert len(LABELS) == 3 and LABELS == ("negative", "neutral", "positive")


def test_stratified_split_proportions():
    rng = np.random.default_rng(0)
    n = 1200
    frame = pd.DataFrame({
        SENTENCE_COL: [f"sentence number {i}" for i in range(n)],
        LABEL_COL: rng.choice(LABELS, size=n, p=[0.6, 0.3, 0.1]),
    })
    config = StudyConfig()
    splits = make_splits(frame, config)
    # sklearn ceils the test count, so allow +/-1 row on each split boundary
    assert abs(splits.sizes["train"] - 840) <= 1
    assert abs(splits.sizes["val"] - 180) <= 1
    assert abs(splits.sizes["test"] - 180) <= 1
    for idx in (splits.train, splits.val, splits.test):
        props = frame.iloc[idx][LABEL_COL].value_counts(normalize=True)
        # n=180 per split: one row is ~0.0056, so 0.03 is ~5 rows of slack
        for cls, target in (("negative", 0.6), ("neutral", 0.3), ("positive", 0.1)):
            assert abs(props.get(cls, 0.0) - target) < 0.03


SPLIT_FIXTURE_CONFIG = StudyConfig(test_size=0.25, val_size=0.25)


def test_split_determinism_same_seed():
    frame = load_synthetic_fixture()
    a = make_splits(frame, StudyConfig(random_state=42, test_size=0.25, val_size=0.25))
    b = make_splits(frame, StudyConfig(random_state=42, test_size=0.25, val_size=0.25))
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)


def test_split_seed_changes_partition():
    frame = load_synthetic_fixture()
    a = make_splits(frame, StudyConfig(random_state=42, test_size=0.25, val_size=0.25))
    b = make_splits(frame, StudyConfig(random_state=43, test_size=0.25, val_size=0.25))
    assert a.test != b.test


# ---------------------------------------------------------------------------
# classical models
# ---------------------------------------------------------------------------
def test_majority_baseline_predicts_most_frequent():
    preds = majority_baseline(["neutral", "neutral", "positive"])
    assert len(preds) == 3 and set(preds) == {"neutral"}


def test_tfidf_logreg_trains_and_predicts(fixture_frame):
    tr = [0, 1, 4, 5, 8, 9]  # 2 sentences per class
    te = [2, 3, 6, 7, 10, 11]  # 2 sentences per class
    model = TfidfLogReg(C=1.0, min_df=1).fit(
        fixture_frame.iloc[tr][SENTENCE_COL].tolist(),
        fixture_frame.iloc[tr][LABEL_COL].tolist(),
    )
    preds = model.predict(fixture_frame.iloc[te][SENTENCE_COL].tolist())
    assert set(preds) <= set(LABELS)
    truth = fixture_frame.iloc[te][LABEL_COL].tolist()
    assert accuracy(truth, preds) >= 0.5


def test_tfidf_proba_rows_sum_to_one(fixture_frame):
    model = TfidfLogReg(C=1.0, min_df=1).fit(
        fixture_frame[SENTENCE_COL].tolist()[:9], fixture_frame[LABEL_COL].tolist()[:9]
    )
    proba = model.predict_proba(fixture_frame[SENTENCE_COL].tolist()[9:])
    assert proba.shape == (3, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_embedding_logreg_with_toy_embedder(fixture_frame):
    tr = [0, 1, 4, 5, 8, 9]  # 2 sentences per class
    te = [2, 3, 6, 7, 10, 11]
    tr_texts = fixture_frame.iloc[tr][SENTENCE_COL].tolist()
    te_texts = fixture_frame.iloc[te][SENTENCE_COL].tolist()
    model = EmbeddingLogReg(toy_embed, C=1.0).fit(
        tr_texts, fixture_frame.iloc[tr][LABEL_COL].tolist()
    )
    preds = model.predict(te_texts)
    assert set(preds) <= set(LABELS)
    truth = fixture_frame.iloc[te][LABEL_COL].tolist()
    assert accuracy(truth, preds) >= 0.5


def test_embedding_logreg_probabilities_aligned(fixture_frame):
    model = EmbeddingLogReg(toy_embed, C=1.0).fit(
        fixture_frame[SENTENCE_COL].tolist(), fixture_frame[LABEL_COL].tolist()
    )
    proba = model.predict_proba(fixture_frame[SENTENCE_COL].tolist()[:4])
    assert proba.shape == (4, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_tune_c_selects_best_validation_macro_f1(fixture_frame):
    tr = [0, 1, 4, 5, 8, 9]
    va = [2, 3, 6, 7, 10, 11]
    texts = fixture_frame[SENTENCE_COL].tolist()
    labels = fixture_frame[LABEL_COL].tolist()
    best_c, records = tune_c_on_validation(
        lambda c: TfidfLogReg(C=c, min_df=1),
        [texts[i] for i in tr], [labels[i] for i in tr],
        [texts[i] for i in va], [labels[i] for i in va],
        c_grid=(0.5, 2.0),
    )
    assert best_c in (0.5, 2.0)
    best_record = max(records, key=lambda r: r["val_macro_f1"])
    assert best_c == best_record["C"]


def test_cv_macro_f1_returns_folds(fixture_frame):
    cv = cv_macro_f1(
        lambda c: TfidfLogReg(C=c, min_df=1), 1.0,
        fixture_frame[SENTENCE_COL].tolist(), fixture_frame[LABEL_COL].tolist(),
        n_folds=3, seed=42,
    )
    assert cv["n_folds"] == 3 and len(cv["fold_scores"]) == 3
    assert 0.0 <= cv["mean_macro_f1"] <= 1.0


# ---------------------------------------------------------------------------
# heavy model smoke tests (skipped on light CI installs)
# ---------------------------------------------------------------------------
def test_sentence_transformer_embeddings_smoke():
    pytest.importorskip("torch")
    st = pytest.importorskip("sentence_transformers")
    from financial_nlp_sentiment.models import make_sentence_embedder

    embed = make_sentence_embedder("sentence-transformers/all-MiniLM-L6-v2")
    vecs = embed(["Operating profit rose.", "The company reported a net loss."])
    assert vecs.shape == (2, 384)
    assert np.isfinite(vecs).all()
    assert st  # silence unused-import warning


def test_finbert_wrapper_returns_valid_probabilities():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from financial_nlp_sentiment.models import finbert_predict

    proba, per_1000 = finbert_predict(
        ["Operating profit rose to 14 million euros.", "The group cut its outlook."],
        batch_size=2,
        max_length=32,
    )
    assert proba.shape == (2, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all()
    assert per_1000 > 0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_macro_f1_matches_hand_computation():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    # class 0: P=1/2, R=1/2 -> F1=1/2; class 1: P=2/3, R=1 -> F1=4/5;
    # class 2: P=1, R=1/2 -> F1=2/3; macro = (1/2 + 4/5 + 2/3)/3 = 59/90
    from financial_nlp_sentiment.config import ID_TO_LABEL

    y_true_lab = [ID_TO_LABEL[i] for i in y_true]
    y_pred_lab = [ID_TO_LABEL[i] for i in y_pred]
    assert macro_f1(y_true_lab, y_pred_lab) == pytest.approx(59 / 90)


def test_confusion_matrix_shape_and_sum():
    truth = ["neutral"] * 6 + ["positive"] * 3 + ["negative"] * 3
    preds = list(majority_baseline(truth))
    assert set(preds) == {"neutral"}
    cm = confusion(truth, preds)
    assert cm.shape == (3, 3)
    assert int(cm.sum()) == len(truth)
    assert cm[1, 1] == 6  # majority class perfectly recovered
    assert cm[0, 1] == 3 and cm[2, 1] == 3  # minority classes swallowed


def test_per_class_metrics_matches_confusion(fixture_frame):
    truth = fixture_frame[LABEL_COL].tolist()[:9]
    preds = fixture_frame[LABEL_COL].tolist()[6:] + fixture_frame[LABEL_COL].tolist()[:3]
    table = per_class_metrics(truth, preds)
    cm = confusion(truth, preds)
    for i, cls in enumerate(LABELS):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        expected_p = tp / (tp + fp) if tp + fp else 0.0
        expected_r = tp / (tp + fn) if tp + fn else 0.0
        assert table.loc[cls, "precision"] == pytest.approx(expected_p, abs=1e-9)
        assert table.loc[cls, "recall"] == pytest.approx(expected_r, abs=1e-9)


def test_multiclass_brier_bounds():
    onehot = np.array([[1, 0, 0], [0, 1, 0]])
    assert multiclass_brier(["negative", "neutral"], onehot) == pytest.approx(0.0)
    uniform = np.full((2, 3), 1 / 3)
    # sum_k (1/3 - y_k)^2 = 4/9 + 1/9 + 1/9 = 2/3 per sample
    assert multiclass_brier(["negative", "neutral"], uniform) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def test_mcnemar_hand_computed():
    # 12 paired outcomes: A right on 10 where B wrong (b=10), A wrong on 2
    # where B right (c=2). Exact two-sided p = 2 * P(X<=2 | n=12, p=0.5).
    y_true = [f"item{i}" for i in range(12)]
    pred_a = [f"item{i}" if i < 10 else "other" for i in range(12)]
    pred_b = ["other" if i < 10 else f"item{i}" for i in range(12)]
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert result["b_a_right_b_wrong"] == 10
    assert result["c_a_wrong_b_right"] == 2
    expected_p = 2 * binom.cdf(2, 12, 0.5)
    assert result["p_value"] == pytest.approx(float(expected_p))
    assert result["p_value"] == pytest.approx(158 / 4096)


def test_mcnemar_symmetric_ties_give_p_one():
    y_true = [f"item{i}" for i in range(6)]
    pred_a = [f"item{i}" if i < 3 else "other" for i in range(6)]
    pred_b = ["other" if i < 3 else f"item{i}" for i in range(6)]
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert result["b_a_right_b_wrong"] == 3
    assert result["c_a_wrong_b_right"] == 3
    assert result["p_value"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# data parsing (offline, no network)
# ---------------------------------------------------------------------------
def test_parse_member_reads_zip_format():
    lines = "\r\n".join([
        "Net sales rose to 10 million euros.@positive",
        "The company was founded in 1998.@neutral",
        "The group posted a loss.@negative",
    ])
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("FinancialPhraseBank-v1.0/Sentences_50Agree.txt", lines)
    frame = _parse_member(buf.getvalue())
    assert list(frame.columns) == [SENTENCE_COL, LABEL_COL]
    assert len(frame) == 3
    assert set(frame[LABEL_COL]) <= set(LABELS)
    assert frame.iloc[0][SENTENCE_COL] == "Net sales rose to 10 million euros."


def test_parse_member_rejects_unknown_label():
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("FinancialPhraseBank-v1.0/Sentences_50Agree.txt", "hello@bogus")
    with pytest.raises(ValueError, match="unexpected label"):
        _parse_member(buf.getvalue())


def test_collect_misclassified_round_robin_pairs():
    texts = [f"sentence {i}" for i in range(12)]
    truth = ["positive"] * 6 + ["negative"] * 6
    pred = ["neutral"] * 12  # two distinct confusion pairs
    out = collect_misclassified(texts, truth, pred, max_examples=4, seed=0)
    assert len(out) == 4
    assert (out["true"] != out["predicted"]).all()
    assert set(out["true"]) == {"positive", "negative"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_study.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "FinBERT" in result.stdout or "FinBERT" in result.stderr
    assert "--seed" in result.stdout
