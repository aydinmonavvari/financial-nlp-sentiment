"""Models for the sentiment classification benchmark.

Five systems, in increasing sophistication:

1. ``majority_baseline``  - always predicts the most frequent training class.
2. VADER                  - general-domain lexicon mapped to 3 classes
                            (``text.vader_predict``; thresholds fixed a priori).
3. ``TfidfLogReg``        - bag-of-words baseline: word 1-2 gram TF-IDF +
                            multinomial logistic regression (C tuned on val).
4. ``EmbeddingLogReg``    - frozen sentence-transformer embeddings
                            (all-MiniLM-L6-v2) + logistic regression.
5. ``finbert_predict``    - domain-specific pretrained model (ProsusAI/finbert),
                            INFERENCE ONLY - no fine-tuning, no training data
                            touching; CPU budget forbids fine-tuning and the
                            experiment is explicitly framed as a reference
                            point, not a clean holdout comparison (possible
                            train/test overlap between FinBERT's corpus and
                            Financial PhraseBank-style text; see README).

All model outputs are label STRINGS from the canonical set
{negative, neutral, positive} so that every metric can be computed uniformly.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

import numpy as np

from .config import LABELS, StudyConfig

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], np.ndarray]


def majority_baseline(train_labels: list[str] | np.ndarray) -> np.ndarray:
    """Predict the most frequent training class for every instance.

    Ties are broken by canonical label order (deterministic).
    """
    counts = Counter(train_labels)
    best = max(LABELS, key=lambda lab: (counts.get(lab, 0), -LABELS.index(lab)))
    return np.array([best] * len(train_labels), dtype=object)


class TfidfLogReg:
    """Word 1-2 gram TF-IDF features + multinomial logistic regression."""

    def __init__(self, C: float = 1.0, seed: int = 42, min_df: int = 2):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.C = C
        self.seed = seed
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=min_df, sublinear_tf=True
        )
        self.clf = LogisticRegression(
            C=C, max_iter=2000, random_state=seed
        )

    def fit(self, texts: list[str], labels: list[str]) -> TfidfLogReg:
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict(self.vectorizer.transform(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        proba = self.clf.predict_proba(self.vectorizer.transform(texts))
        return self._reorder(proba)

    def _reorder(self, proba: np.ndarray) -> np.ndarray:
        """Align probability columns to the canonical LABELS order."""
        mapping = {name: i for i, name in enumerate(self.clf.classes_)}
        idx = [mapping[lab] for lab in LABELS]
        return proba[:, idx]


class EmbeddingLogReg:
    """Frozen sentence embeddings + multinomial logistic regression.

    ``embed_fn`` maps a list of sentences to a 2-D float array (n, dim). In
    the study this is the frozen all-MiniLM-L6-v2 sentence transformer; in
    tests a deterministic toy embedder can be injected (no model download).
    """

    def __init__(self, embed_fn: EmbedFn, C: float = 1.0, seed: int = 42):
        from sklearn.linear_model import LogisticRegression

        self.embed_fn = embed_fn
        self.C = C
        self.clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)

    def fit(self, texts: list[str], labels: list[str]) -> EmbeddingLogReg:
        X = np.asarray(self.embed_fn(texts), dtype=float)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict(np.asarray(self.embed_fn(texts), dtype=float))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        proba = self.clf.predict_proba(np.asarray(self.embed_fn(texts), dtype=float))
        mapping = {name: i for i, name in enumerate(self.clf.classes_)}
        idx = [mapping[lab] for lab in LABELS]
        return proba[:, idx]


def make_sentence_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
) -> EmbedFn:
    """Return a callable producing frozen sentence-transformer embeddings.

    Heavy imports (torch / sentence-transformers) live INSIDE this factory so
    that importing this module never requires them (CI runs base deps only).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

    return embed


def finbert_predict(
    texts: list[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 64,
    max_length: int = 64,
) -> tuple[np.ndarray, float]:
    """FinBERT inference-only predictions aligned to canonical LABELS order.

    Returns
    -------
    (proba, seconds_per_1000):
        ``proba`` has shape (n, 3) with columns ordered
        [negative, neutral, positive]; ``seconds_per_1000`` is the measured
        CPU throughput, recorded for the runtime budget documentation.

    Notes
    -----
    ProsusAI/finbert was fine-tuned by its authors on financial text including
    Financial PhraseBank-style data; its scores are treated here as a
    domain-model REFERENCE, not a clean holdout result. No fine-tuning is
    performed in this project (CPU budget; inference only).
    """
    import time

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    probas: list[np.ndarray] = []
    started = time.perf_counter()
    n_done = 0
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            enc = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probas.append(torch.softmax(logits, dim=-1).cpu().numpy())
            n_done += len(chunk)
    elapsed = time.perf_counter() - started
    per_1000 = elapsed / max(n_done, 1) * 1000.0

    proba = np.vstack(probas)
    # FinBERT's own class order differs from the canonical order - remap by name.
    finbert_labels = [model.config.id2label[i].lower() for i in range(proba.shape[1])]
    idx = [finbert_labels.index(lab) for lab in LABELS]
    return proba[:, idx], per_1000


def tune_c_on_validation(
    make_model: Callable[[float], object],
    train_texts: list[str],
    train_labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    c_grid: tuple[float, ...],
) -> tuple[float, list[dict]]:
    """Pick C by validation macro-F1; return (best_C, per-C records)."""
    from sklearn.metrics import f1_score

    records = []
    for c in c_grid:
        model = make_model(c)
        model.fit(train_texts, train_labels)
        preds = model.predict(val_texts)
        macro = float(f1_score(val_labels, preds, average="macro"))
        records.append({"C": c, "val_macro_f1": round(macro, 6)})
        logger.info("C=%.2f -> val macro-F1 %.4f", c, macro)
    best = max(records, key=lambda r: (r["val_macro_f1"], -r["C"]))
    return float(best["C"]), records


def cv_macro_f1(
    make_model: Callable[[float], object],
    C: float,
    texts: list[str],
    labels: list[str],
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Stratified k-fold CV macro-F1 on the training split (classical models)."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for train_idx, hold_idx in skf.split(texts, labels):
        fold_train = [texts[i] for i in train_idx]
        fold_hold = [texts[i] for i in hold_idx]
        model = make_model(C)
        model.fit(fold_train, [labels[i] for i in train_idx])
        preds = model.predict(fold_hold)
        scores.append(float(f1_score([labels[i] for i in hold_idx], preds, average="macro")))
    return {
        "n_folds": n_folds,
        "C": C,
        "mean_macro_f1": round(float(np.mean(scores)), 6),
        "std_macro_f1": round(float(np.std(scores)), 6),
        "fold_scores": [round(s, 6) for s in scores],
    }


def default_config_note(config: StudyConfig) -> dict:
    """Machine-readable record of the a priori analysis choices."""
    return {
        "vader_pos_threshold": config.vader_pos_threshold,
        "vader_neg_threshold": config.vader_neg_threshold,
        "c_grid": list(config.c_grid),
        "n_folds": config.n_folds,
        "seed": config.random_state,
        "embedding_model": config.embedding_model,
        "finbert_model": config.finbert_model,
        "finbert_batch_size": config.finbert_batch_size,
        "finbert_max_length": config.finbert_max_length,
    }


__all__ = [
    "EmbeddingLogReg",
    "TfidfLogReg",
    "cv_macro_f1",
    "default_config_note",
    "finbert_predict",
    "majority_baseline",
    "make_sentence_embedder",
    "tune_c_on_validation",
]
