"""Configuration for the financial sentiment benchmark study.

Scope note: this project is a *classification benchmark* on the Financial
PhraseBank corpus. It measures how well different sentiment-classification
approaches label financial-news sentences. It makes no claims about trading
profitability or market prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical label order used everywhere in this package. The Financial
# PhraseBank files carry string labels; every model output is mapped back to
# this fixed order so confusion matrices and probabilities stay comparable.
LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
LABEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = {i: name for name, i in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class StudyConfig:
    """Static configuration for one run of the benchmark study.

    Attributes
    ----------
    val_size, test_size:
        Fractions of the corpus held out (stratified by label) for the
        validation split (model selection: C tuning) and the untouched test
        split. The remainder is the training split. 70/15/15 in the base case.
    random_state:
        Global seed; the split and every model pin their own seed to it.
    n_folds:
        Number of stratified CV folds computed on the training split for the
        two classical models (TF-IDF and embeddings).
    vader_pos_threshold, vader_neg_threshold:
        Compound-score cut-offs that map VADER output to the 3-class problem.
        Chosen a priori (before looking at any Financial PhraseBank output):
        compound >= 0.35 -> positive, compound <= -0.15 -> negative, else
        neutral. The asymmetry is deliberate: VADER's general-domain lexicon
        scores mildly positive finance prose (percentages, growth verbs)
        positively, so the neutral band is wide. These thresholds are NOT
        tuned on any split and this is documented as a design decision.
    c_grid:
        Regularization strengths tuned for the two logistic-regression models
        on the validation split (macro-F1).
    finbert_model:
        Hugging Face id of the domain-specific pretrained model, used for
        INFERENCE ONLY (no fine-tuning; CPU budget).
    finbert_batch_size, finbert_max_length:
        Inference throughput controls for FinBERT on CPU.
    """

    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    n_folds: int = 5

    vader_pos_threshold: float = 0.35
    vader_neg_threshold: float = -0.15

    c_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    finbert_model: str = "ProsusAI/finbert"
    finbert_batch_size: int = 64
    finbert_max_length: int = 64

    hf_repo: str = "takala/financial_phrasebank"
    hf_filename: str = "data/FinancialPhraseBank-v1.0.zip"
    phrasebank_config: str = "sentences_50agree"

    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    figures_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "figures")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.figures_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


def sanitize_config_paths(values: dict) -> dict:
    """Make a serialized config portable: ``Path`` values -> posix strings.

    The serialized config is embedded verbatim in ``reports/metrics.json``, which
    is committed to the repository. Absolute, machine-specific paths must never
    leak into it: any ``Path`` that lives under ``PROJECT_ROOT``
    is stored as a ``/``-separated path *relative to the project root*
    (e.g. ``"data/raw"``). Paths outside the project root (explicit user
    overrides) are kept as absolute posix strings — portable where possible,
    honest where not.
    """
    out: dict = {}
    for key, value in values.items():
        if isinstance(value, Path):
            try:
                value = value.resolve().relative_to(PROJECT_ROOT)
            except ValueError:
                pass
            out[key] = value.as_posix()
        else:
            out[key] = value
    return out


DEFAULT_CONFIG = StudyConfig()
