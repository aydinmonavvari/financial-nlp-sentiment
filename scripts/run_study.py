#!/usr/bin/env python
"""CLI for the financial sentiment benchmark study.

Examples
--------
    python scripts/run_study.py               # full benchmark (downloads on first run)
    python scripts/run_study.py --help        # show options
    python scripts/run_study.py --no-figures  # metrics + reports only
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_nlp_sentiment.config import DEFAULT_CONFIG  # noqa: E402
from financial_nlp_sentiment.pipeline import run_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_study",
        description=(
            "Benchmark five sentiment-classification approaches (majority, VADER "
            "lexicon, TF-IDF+logistic regression, frozen MiniLM embeddings+logistic "
            "regression, FinBERT inference) on the Financial PhraseBank corpus. "
            "Classification benchmark only - no trading or market-prediction claims."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_CONFIG.random_state,
        help="global seed for splits and models (default: 42)",
    )
    parser.add_argument(
        "--test-size", type=float, default=DEFAULT_CONFIG.test_size,
        help="test split fraction (default: 0.15)",
    )
    parser.add_argument(
        "--val-size", type=float, default=DEFAULT_CONFIG.val_size,
        help="validation split fraction (default: 0.15)",
    )
    parser.add_argument(
        "--finbert-batch-size", type=int, default=DEFAULT_CONFIG.finbert_batch_size,
        help="FinBERT inference batch size (default: 64)",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="skip figure generation (metrics + reports only)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = replace(
        DEFAULT_CONFIG,
        random_state=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        finbert_batch_size=args.finbert_batch_size,
    )
    out = run_pipeline(config, save_outputs=not args.no_figures)
    results = out["results"]
    best = results.sort_values("macro_f1", ascending=False).iloc[0]
    print("\n=== Test-set leaderboard (macro-F1) ===")
    print(results.to_string(index=False))
    print(f"\nBest system: {best['model']} (macro-F1 {best['macro_f1']:.4f})")
    print("Reports: reports/metrics.json, reports/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
