"""Minimal text preprocessing and the VADER lexicon baseline.

Preprocessing policy (deliberate and documented):
- The ONLY normalization is whitespace collapsing (``normalize_text``).
- Case is PRESERVED everywhere. VADER's lexicon is case-sensitive by design
  (ALL-CAPS words amplify polarity), and the TF-IDF vectorizer lowercases
  internally (sklearn default), which keeps the two code paths explicit
  rather than hiding a destructive transform at the data layer.
- No stemming, stopword removal or punctuation stripping: the corpus is
  short, well-formed financial news sentences; aggressive cleaning removes
  the very signals (negation, ``%`` tokens) this benchmark studies.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse internal whitespace runs to single spaces and strip ends."""
    return _WS.sub(" ", str(text)).strip()


def normalize_texts(texts: Iterable[str]) -> list[str]:
    return [normalize_text(t) for t in texts]


def vader_compound_scores(texts: Iterable[str]) -> np.ndarray:
    """VADER compound scores in [-1, 1] for each text (general-domain lexicon).

    VADER (Hutto & Gilbert 2014) is a rule/lexicon model built for social
    media text; it is used here unchanged as a general-domain baseline. Its
    known finance limitation is discussed in the README/report: it has no
    finance lexicon (no treatment of e.g. "downgrade", "profit warning" as a
    construct) and its neutral band mis-calibrates on terse factual finance
    prose. See ``vader_to_label`` for the fixed threshold mapping.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return np.array(
        [analyzer.polarity_scores(normalize_text(t))["compound"] for t in texts],
        dtype=float,
    )


def vader_to_label(
    compound: float,
    pos_threshold: float = 0.35,
    neg_threshold: float = -0.15,
) -> str:
    """Map one VADER compound score to the 3-class problem.

    Thresholds are FIXED A PRIORI (chosen before any PhraseBank output was
    inspected; not tuned on any split):
        compound >= pos_threshold  -> "positive"
        compound <= neg_threshold  -> "negative"
        else                       -> "neutral"

    The asymmetric band is deliberate: VADER scores mildly positive finance
    prose (growth verbs, percentage gains) positively, so the neutral band
    must be wider on the positive side to keep terse factual sentences out of
    the positive class. The exact values used in the study are recorded in
    ``StudyConfig`` and reported in the README.
    """
    if compound >= pos_threshold:
        return "positive"
    if compound <= neg_threshold:
        return "negative"
    return "neutral"


def vader_predict(
    texts: Iterable[str],
    pos_threshold: float = 0.35,
    neg_threshold: float = -0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: compound scores + predicted labels for a text batch."""
    compounds = vader_compound_scores(texts)
    labels = np.array(
        [vader_to_label(c, pos_threshold, neg_threshold) for c in compounds],
        dtype=object,
    )
    return compounds, labels


__all__ = [
    "normalize_text",
    "normalize_texts",
    "vader_compound_scores",
    "vader_predict",
    "vader_to_label",
]
