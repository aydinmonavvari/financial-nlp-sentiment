"""Data loading for the Financial PhraseBank corpus (Malo et al. 2014).

Provenance: the corpus is distributed as the ``FinancialPhraseBank-v1.0`` zip
archive, mirrored on the Hugging Face Hub under the dataset id
``takala/financial_phrasebank``. This module downloads the archive once,
caches it under ``data/raw/`` (git-ignored) and parses the
``Sentences_50Agree.txt`` member (lines formatted ``sentence@label``).

Encoding: the member files are ISO-8859-1 (verified byte-level: a strict UTF-8
decode fails, and the archive contains ~160 non-ASCII bytes that are valid
latin-1 accents such as n-tilde and a-grave). The loader therefore decodes with
``iso-8859-1`` and never edits corpus text. Note that the upstream archive
itself contains pre-corrupted accented characters (e.g. the byte sequence that
latin-1-decodes to ``Bj+Ârn`` where ``Björn`` was intended, plus ~134 similar
``+<accent>`` artifacts in ``Sentences_50Agree.txt``, present in all four
member files of the official v1.0 distribution). These upstream artifacts are
preserved verbatim — re-encoding or "repairing" them would mean fabricating
corpus text; a UTF-8 decode is impossible and would additionally destroy the
151 legitimate accented characters (see README §7 data provenance).

The corpus is NEVER committed to this repository.
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import LABELS, PROJECT_ROOT, StudyConfig

logger = logging.getLogger(__name__)

SENTENCE_COL = "sentence"
LABEL_COL = "label"


@dataclass(frozen=True)
class Splits:
    """Index splits into the (deduplicated) corpus frame."""

    train: list[int]
    val: list[int]
    test: list[int]

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _parse_member(
    archive_bytes: bytes, member_suffix: str = "Sentences_50Agree.txt"
) -> pd.DataFrame:
    """Parse a ``sentence@label`` text member of the PhraseBank zip archive.

    The member is decoded as ISO-8859-1 (the encoding of the official
    distribution; see the module docstring for the byte-level evidence and the
    note on pre-corrupted accented characters inherited from upstream).
    """
    rows: list[tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        member = next(n for n in zf.namelist() if n.endswith(member_suffix))
        text = zf.read(member).decode("iso-8859-1")
    for line in text.splitlines():
        line = line.strip()
        if not line or "@" not in line:
            continue
        sentence, _, label = line.rpartition("@")
        label = label.strip().lower()
        if label not in LABELS:
            raise ValueError(f"unexpected label {label!r} in PhraseBank archive")
        rows.append((sentence.strip(), label))
    frame = pd.DataFrame(rows, columns=[SENTENCE_COL, LABEL_COL])
    return frame


def load_phrasebank(config: StudyConfig | None = None, offline: bool = False) -> pd.DataFrame:
    """Load the ``sentences_50agree`` configuration of Financial PhraseBank.

    Downloads and caches the upstream archive on first call; later calls read
    from the local cache. Exact-duplicate sentence texts are dropped (keeping
    the first occurrence) BEFORE splitting, so no sentence can leak across
    splits. This affects a handful of rows and is reported in the logs; see
    README section on data provenance.

    Returns
    -------
    pd.DataFrame with columns ``sentence`` (str) and ``label`` (str, one of
    negative/neutral/positive), deduplicated, original corpus order.
    """
    config = config or StudyConfig()
    config.ensure_dirs()
    zip_path = config.raw_dir / "FinancialPhraseBank-v1.0.zip"
    if not zip_path.exists():
        if offline:
            raise FileNotFoundError(
                f"cached corpus not found at {zip_path}; run once with network access"
            )
        from huggingface_hub import hf_hub_download

        logger.info("downloading %s :: %s", config.hf_repo, config.hf_filename)
        downloaded = hf_hub_download(
            config.hf_repo, config.hf_filename, repo_type="dataset"
        )
        zip_path.write_bytes(Path(downloaded).read_bytes())
        logger.info("cached archive at %s", zip_path)
    frame = _parse_member(zip_path.read_bytes())
    n_before = len(frame)
    frame = frame.drop_duplicates(subset=[SENTENCE_COL], keep="first").reset_index(drop=True)
    # Persist the pre-dedup row count on the frame so the pipeline can report
    # both sides of the deduplication step in reports/metrics.json.
    frame.attrs["n_rows_before_dedup"] = n_before
    logger.info(
        "phrasebank[%s]: %d rows -> %d unique sentences (%d duplicates dropped)",
        config.phrasebank_config,
        n_before,
        len(frame),
        n_before - len(frame),
    )
    return frame


def label_distribution(frame: pd.DataFrame) -> pd.Series:
    """Class distribution as proportions, ordered negative/neutral/positive."""
    counts = frame[LABEL_COL].value_counts().reindex(LABELS, fill_value=0)
    return counts / counts.sum()


def make_splits(frame: pd.DataFrame, config: StudyConfig | None = None) -> Splits:
    """Stratified 70/15/15 split (train/validation/test) with a fixed seed.

    A stratified (label-random) split is appropriate here by design: the
    corpus is an unordered pool of standalone sentences without timestamps,
    so there is no temporal ordering to preserve. Randomly holding out
    sentences keeps the label distribution identical across splits, which is
    the comparison this benchmark needs. The split is NOT time series data
    and no market-prediction claim is made anywhere in this project.
    """
    from sklearn.model_selection import train_test_split

    config = config or StudyConfig()
    y = frame[LABEL_COL]
    idx_trainval, idx_test = train_test_split(
        range(len(frame)),
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )
    # val fraction is relative to the trainval remainder (0.15 / 0.85)
    rel_val = config.val_size / (1.0 - config.test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=rel_val,
        stratify=y.iloc[idx_trainval],
        random_state=config.random_state,
    )
    splits = Splits(
        train=sorted(idx_train), val=sorted(idx_val), test=sorted(idx_test)
    )
    logger.info(
        "stratified splits: %s (seed=%d)",
        splits.sizes,
        config.random_state,
    )
    return splits


def save_split_assignment(frame: pd.DataFrame, splits: Splits, config: StudyConfig) -> None:
    """Persist which split each sentence landed in (audit artifact)."""
    config.ensure_dirs()
    out = frame[[SENTENCE_COL, LABEL_COL]].copy()
    assignment = pd.Series("train", index=out.index, name="split")
    assignment.loc[splits.val] = "val"
    assignment.loc[splits.test] = "test"
    out["split"] = assignment
    path = config.processed_dir / "split_assignment.parquet"
    out.to_parquet(path, index=False)
    logger.info("split assignment written to %s", path)


def load_synthetic_fixture(n_per_class: int = 4) -> pd.DataFrame:
    """Tiny synthetic labeled sentences for offline tests only.

    These sentences are written by hand for this repository (clearly synthetic
    test fixtures, NOT research data) so the test suite never needs the
    network or the real corpus.
    """
    texts = [
        "Operating profit rose to 14 million euros from 9 million euros a year earlier.",
        "The company raised its full-year guidance after a strong third quarter.",
        "Net sales increased 12 percent year on year, beating market expectations.",
        "The group returned to profitability in the second quarter of the year.",
        "The group reported a net loss of 5 million euros for the period.",
        "Operating profit fell sharply after one-off restructuring costs.",
        "The company lowered its outlook for the coming fiscal year.",
        "Sales decreased by 8 percent and the order backlog continued to shrink.",
        "The company was founded in 1998 and is headquartered in Helsinki.",
        "The share was traded at 12.50 euros on the Nasdaq Helsinki.",
        "The board will convene on April 15 to review the annual report.",
        "The company employs 450 people at its production plant in Tampere.",
    ]
    labels = (
        ["positive"] * n_per_class + ["negative"] * n_per_class + ["neutral"] * n_per_class
    )
    return pd.DataFrame({SENTENCE_COL: texts, LABEL_COL: labels})


__all__ = [
    "LABEL_COL",
    "PROJECT_ROOT",
    "SENTENCE_COL",
    "Splits",
    "label_distribution",
    "load_phrasebank",
    "load_synthetic_fixture",
    "make_splits",
    "save_split_assignment",
]
