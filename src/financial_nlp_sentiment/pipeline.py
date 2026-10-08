"""End-to-end orchestration for the financial sentiment benchmark study.

Scope: a CLASSIFICATION BENCHMARK of five sentiment-labeling approaches on
the Financial PhraseBank corpus. This pipeline computes label-level metrics
only. It does NOT simulate trading, does NOT touch price data and makes no
market-prediction claims.
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ID_TO_LABEL, LABELS, StudyConfig, sanitize_config_paths
from .data import (
    LABEL_COL,
    SENTENCE_COL,
    label_distribution,
    load_phrasebank,
    make_splits,
    save_split_assignment,
)
from .evaluation import (
    accuracy,
    collect_misclassified,
    confusion,
    macro_f1,
    mcnemar_test,
    mean_confidence,
    multiclass_brier,
    per_class_metrics,
)
from .models import (
    EmbeddingLogReg,
    TfidfLogReg,
    cv_macro_f1,
    default_config_note,
    finbert_predict,
    majority_baseline,
    make_sentence_embedder,
    tune_c_on_validation,
)
from .text import vader_predict

logger = logging.getLogger(__name__)

MODEL_ORDER = (
    "majority",
    "vader_lexicon",
    "tfidf_logreg",
    "minilm_embeddings_logreg",
    "finbert_inference",
)

MODEL_LABELS = {
    "majority": "Majority",
    "vader_lexicon": "VADER",
    "tfidf_logreg": "TF-IDF+LR",
    "minilm_embeddings_logreg": "MiniLM+LR",
    "finbert_inference": "FinBERT",
}


def run_pipeline(config: StudyConfig | None = None, save_outputs: bool = True) -> dict:
    config = config or StudyConfig()
    config.ensure_dirs()
    t_wall = time.perf_counter()
    timings: dict[str, float] = {}

    # ---- 1) data -----------------------------------------------------------
    t0 = time.perf_counter()
    frame = load_phrasebank(config)
    splits = make_splits(frame, config)
    if save_outputs:
        save_split_assignment(frame, splits, config)
    timings["data"] = round(time.perf_counter() - t0, 2)

    texts = frame[SENTENCE_COL].tolist()
    labels = frame[LABEL_COL].tolist()
    tr, va, te = splits.train, splits.val, splits.test
    train_texts = [texts[i] for i in tr]
    val_texts = [texts[i] for i in va]
    test_texts = [texts[i] for i in te]
    train_labels = [labels[i] for i in tr]
    val_labels = [labels[i] for i in va]
    test_labels = [labels[i] for i in te]
    seed = config.random_state
    logger.info("train=%d val=%d test=%d", len(tr), len(va), len(te))

    dataset_block = {
        "hf_repo": config.hf_repo,
        "archive": "FinancialPhraseBank-v1.0.zip (upstream distribution)",
        "config_name": config.phrasebank_config,
        "n_rows_before_dedup": int(frame.attrs.get("n_rows_before_dedup", len(frame))),
        "n_rows_after_dedup": len(frame),
        "text_encoding": "ISO-8859-1 (byte-level verified; decoded faithfully, never re-encoded)",
        "encoding_note": (
            "The upstream archive itself contains pre-corrupted accented "
            "characters (e.g. latin-1 bytes rendering as 'Bj+\u00c2rn' where "
            "'Bj\u00f6rn' was intended; ~134 '+<accent>' artifacts in "
            "Sentences_50Agree.txt, present in all four member files of the "
            "official v1.0 distribution). The loader preserves corpus text "
            "verbatim; see README data provenance for the full diagnosis."
        ),
        "class_counts": {k: int(v) for k, v in frame[LABEL_COL].value_counts().items()},
        "class_proportions": {
            k: round(float(v), 4) for k, v in label_distribution(frame).items()
        },
        "split_sizes": splits.sizes,
        "split_class_proportions": {
            name: {
                k: round(float(v), 4)
                for k, v in label_distribution(frame.iloc[idxs]).items()
            }
            for name, idxs in (("train", tr), ("val", va), ("test", te))
        },
        "split_design_note": (
            "Stratified label-random 70/15/15 split (seed=42); the corpus is an "
            "unordered pool of standalone sentences with no timestamps, so a "
            "time-based split is not applicable and no market-prediction claim "
            "is made."
        ),
    }

    # ---- 2) majority baseline ----------------------------------------------
    t0 = time.perf_counter()
    majority_test = list(majority_baseline(train_labels)[: len(test_labels)])
    timings["majority"] = round(time.perf_counter() - t0, 4)

    # ---- 3) VADER lexicon baseline (thresholds fixed a priori) --------------
    t0 = time.perf_counter()
    compounds, vader_labels_all = vader_predict(
        texts,
        pos_threshold=config.vader_pos_threshold,
        neg_threshold=config.vader_neg_threshold,
    )
    timings["vader"] = round(time.perf_counter() - t0, 2)

    # ---- 4) TF-IDF + logistic regression ------------------------------------
    t0 = time.perf_counter()
    best_c_tfidf, tfidf_records = tune_c_on_validation(
        lambda c: TfidfLogReg(C=c, seed=seed),
        train_texts, train_labels, val_texts, val_labels, config.c_grid,
    )
    tfidf_model = TfidfLogReg(C=best_c_tfidf, seed=seed).fit(train_texts, train_labels)
    tfidf_test = list(tfidf_model.predict(test_texts))
    tfidf_proba = tfidf_model.predict_proba(test_texts)
    tfidf_cv = cv_macro_f1(
        lambda c: TfidfLogReg(C=c, seed=seed), best_c_tfidf,
        train_texts, train_labels, n_folds=config.n_folds, seed=seed,
    )
    timings["tfidf"] = round(time.perf_counter() - t0, 2)

    # ---- 5) frozen embeddings + logistic regression -------------------------
    t0 = time.perf_counter()
    embed = make_sentence_embedder(config.embedding_model)
    embeddings_all = np.asarray(embed(texts), dtype=float)
    timings["embedding_corpus"] = round(time.perf_counter() - t0, 2)
    index_of_text = {t: i for i, t in enumerate(texts)}

    def embed_lookup(batch: list[str]) -> np.ndarray:
        return embeddings_all[[index_of_text[t] for t in batch]]

    t0 = time.perf_counter()
    best_c_emb, emb_records = tune_c_on_validation(
        lambda c: EmbeddingLogReg(embed_lookup, C=c, seed=seed),
        train_texts, train_labels, val_texts, val_labels, config.c_grid,
    )
    emb_model = EmbeddingLogReg(embed_lookup, C=best_c_emb, seed=seed).fit(
        train_texts, train_labels
    )
    emb_test = list(emb_model.predict(test_texts))
    emb_proba = emb_model.predict_proba(test_texts)
    emb_cv = cv_macro_f1(
        lambda c: EmbeddingLogReg(embed_lookup, C=c, seed=seed), best_c_emb,
        train_texts, train_labels, n_folds=config.n_folds, seed=seed,
    )
    timings["embedding_logreg"] = round(time.perf_counter() - t0, 2)

    # ---- 6) FinBERT (inference only, test split) ----------------------------
    t0 = time.perf_counter()
    finbert_proba, per_1000 = finbert_predict(
        test_texts,
        model_name=config.finbert_model,
        batch_size=config.finbert_batch_size,
        max_length=config.finbert_max_length,
    )
    finbert_test = [ID_TO_LABEL[i] for i in finbert_proba.argmax(axis=1)]
    finbert_seconds = round(time.perf_counter() - t0, 2)
    timings["finbert_test_inference"] = finbert_seconds

    pred_map = {
        "majority": majority_test,
        "vader_lexicon": list(vader_labels_all[te]),
        "tfidf_logreg": tfidf_test,
        "minilm_embeddings_logreg": emb_test,
        "finbert_inference": finbert_test,
    }
    proba_map = {
        "tfidf_logreg": tfidf_proba,
        "minilm_embeddings_logreg": emb_proba,
        "finbert_inference": finbert_proba,
    }

    # ---- 7) metrics table ---------------------------------------------------
    def score_system(name: str) -> tuple[dict, pd.DataFrame]:
        preds = pred_map[name]
        per_class = per_class_metrics(test_labels, preds)
        row = {
            "model": name,
            "n_test": len(test_labels),
            "accuracy": round(accuracy(test_labels, preds), 4),
            "macro_f1": round(macro_f1(test_labels, preds), 4),
            "per_class": {
                cls: {
                    "precision": round(float(per_class.loc[cls, "precision"]), 4),
                    "recall": round(float(per_class.loc[cls, "recall"]), 4),
                    "f1": round(float(per_class.loc[cls, "f1"]), 4),
                    "support": int(per_class.loc[cls, "support"]),
                }
                for cls in LABELS
            },
        }
        proba = proba_map.get(name)
        if proba is not None:
            row["brier_multiclass"] = round(multiclass_brier(test_labels, proba), 4)
            row["mean_confidence"] = round(mean_confidence(proba), 4)
        return row, per_class

    rows, per_class_tables = [], {}
    for name in MODEL_ORDER:
        row, per_class = score_system(name)
        rows.append(row)
        per_class_tables[name] = per_class
    results = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "per_class"} for r in rows]
    )
    logger.info("\n%s", results.to_string(index=False))

    # ---- 8) McNemar between the two best systems on test --------------------
    ranked = results.sort_values("macro_f1", ascending=False)
    best_two = (ranked.iloc[0]["model"], ranked.iloc[1]["model"])
    mcnemar = {
        "system_a": best_two[0],
        "system_b": best_two[1],
        **mcnemar_test(test_labels, pred_map[best_two[0]], pred_map[best_two[1]]),
    }
    mcnemar["verdict"] = (
        "significant at alpha=0.05"
        if mcnemar["p_value"] < 0.05
        else "NOT significant at alpha=0.05"
    )
    mcnemar["interpretation"] = (
        "two-sided exact McNemar on paired test correctness; "
        f"p={mcnemar['p_value']:.3e} -> {mcnemar['verdict']}"
    )

    # ---- 9) error analysis (best trainable system) --------------------------
    trainable = results[
        results["model"].isin(["tfidf_logreg", "minilm_embeddings_logreg"])
    ]
    err_model = trainable.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    test_pos = {t: i for i, t in enumerate(test_texts)}
    errors = collect_misclassified(
        test_texts, test_labels, pred_map[err_model], max_examples=6, seed=seed
    )
    errors["vader_pred"] = [
        pred_map["vader_lexicon"][test_pos[s]] for s in errors["sentence"]
    ]
    errors["finbert_pred"] = [
        pred_map["finbert_inference"][test_pos[s]] for s in errors["sentence"]
    ]
    errors.attrs["model"] = err_model

    # ---- 10) figures + reports ----------------------------------------------
    if save_outputs:
        _make_figures(
            config, frame, test_labels, rows, per_class_tables, pred_map,
            best_two, errors, embeddings_all, labels, compounds,
        )
    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "project": "financial-nlp-sentiment",
        "scope_note": (
            "Classification benchmark on Financial PhraseBank (Malo et al. 2014). "
            "Label-level metrics only; no trading or market-prediction claims."
        ),
        "config": {**sanitize_config_paths(asdict(config)), **default_config_note(config)},
        "dataset": dataset_block,
        "timings_seconds": timings,
        "tuning": {
            "tfidf_logreg": {"best_C": best_c_tfidf, "grid_records": tfidf_records},
            "minilm_embeddings_logreg": {
                "best_C": best_c_emb, "grid_records": emb_records,
            },
        },
        "cv_train": {"tfidf_logreg": tfidf_cv, "minilm_embeddings_logreg": emb_cv},
        "finbert": {
            "model": config.finbert_model,
            "mode": "inference only (no fine-tuning)",
            "coverage": "test split only (documented; directly comparable on test)",
            "n_evaluated": len(test_texts),
            "batch_size": config.finbert_batch_size,
            "max_length": config.finbert_max_length,
            "seconds_per_1000_sentences": round(per_1000, 2),
            "total_seconds": finbert_seconds,
            "caveat": (
                "ProsusAI/finbert's training corpus may overlap with "
                "Financial PhraseBank-style data; treat its scores as a "
                "domain-model reference, not a clean holdout result."
            ),
        },
        "results": rows,
        "mcnemar": mcnemar,
        "error_analysis_model": err_model,
        "error_analysis_examples": errors.to_dict("records"),
        "total_wall_seconds": round(time.perf_counter() - t_wall, 2),
    }
    if save_outputs:
        _save_reports(config, payload, results)

    return {"payload": payload, "results": results, "errors": errors}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _make_figures(config, frame, test_labels, rows, per_class_tables, pred_map,
                  best_two, errors, embeddings_all, labels, compounds) -> None:
    from sklearn.decomposition import PCA

    fig_dir = config.figures_dir
    model_order = [r["model"] for r in rows]

    # 1) class distribution ---------------------------------------------------
    counts = frame[LABEL_COL].value_counts().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(list(LABELS), counts.values, color=["#c44e52", "#8c8c8c", "#4c72b0"])
    ax.bar_label(bars, fmt="%d")
    for i in range(len(LABELS)):
        ax.text(i, counts.values[i] * 0.45, f"{counts.values[i] / counts.sum():.1%}",
                ha="center", color="white", fontsize=11)
    ax.set_title("Financial PhraseBank (sentences_50agree) - class distribution")
    ax.set_ylabel("sentences")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_class_distribution.png", dpi=150)
    plt.close(fig)

    # 2) confusion matrices for the two best systems ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, name in zip(axes, best_two, strict=True):
        cm = confusion(test_labels, pred_map[name])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), [m.title() for m in LABELS], rotation=30)
        ax.set_yticks(range(3), [m.title() for m in LABELS])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{MODEL_LABELS[name]} - confusion (test)")
        thresh = cm.max() / 2 if cm.max() else 0.5
        for r in range(3):
            for c in range(3):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > thresh else "black", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.85)
    fig.suptitle("Confusion matrices - two best systems by test macro-F1")
    fig.savefig(fig_dir / "02_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) per-class F1 across models --------------------------------------------
    supports = per_class_tables[model_order[0]]["support"].astype(int)
    x = np.arange(3)
    width = 0.16
    fig, ax = plt.subplots(figsize=(10, 4.6))
    palette = ["#555555", "#dd8452", "#55a868", "#4c72b0", "#c44e52"]
    for j, name in enumerate(model_order):
        f1s = [per_class_tables[name].loc[cls, "f1"] for cls in LABELS]
        ax.bar(x + (j - 2) * width, f1s, width, label=MODEL_LABELS[name],
               color=palette[j % len(palette)])
    ax.set_xticks(x, [f"{cls.title()}\n(n={supports[cls]})" for cls in LABELS])
    ax.set_ylabel("F1 (test)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1 across systems (test set) - class imbalance view")
    ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(fig_dir / "03_per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4) PCA scatter of frozen embeddings ---------------------------------------
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings_all)
    colors = {"negative": "#c44e52", "neutral": "#8c8c8c", "positive": "#4c72b0"}
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for cls in LABELS:
        mask = np.array([lab == cls for lab in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.45,
                   c=colors[cls], label=f"{cls} (n={mask.sum()})")
    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} var)")
    ax.set_title("Frozen all-MiniLM-L6-v2 embeddings, PCA colored by true label")
    ax.legend(markerscale=2.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_embedding_pca.png", dpi=150)
    plt.close(fig)

    # 5) error-analysis table ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 0.55 * max(len(errors), 1) + 1.4))
    ax.axis("off")
    if len(errors):
        cells = []
        for _, row in errors.iterrows():
            cells.append([
                textwrap.fill(row["sentence"], 62),
                row["true"], row["predicted"], row["vader_pred"], row["finbert_pred"],
            ])
        table = ax.table(
            cellText=cells,
            colLabels=["sentence (true != predicted)", "true", "pred", "VADER", "FinBERT"],
            colWidths=[0.62, 0.095, 0.095, 0.095, 0.095],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 2.0)
        ax.set_title(
            f"Error analysis - {len(errors)} misclassified test sentences "
            f"(system: {MODEL_LABELS[errors.attrs.get('model', 'best trainable')]})",
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(fig_dir / "05_error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6) model comparison bars ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(model_order))
    macro = [r["macro_f1"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    ax.barh(y + 0.2, acc, 0.38, label="accuracy", color="#4c72b0")
    ax.barh(y - 0.2, macro, 0.38, label="macro-F1", color="#dd8452")
    ax.set_yticks(y, [MODEL_LABELS[m] for m in model_order])
    ax.invert_yaxis()
    for yi, (a, f) in enumerate(zip(acc, macro, strict=True)):
        ax.text(a + 0.01, yi + 0.2, f"{a:.3f}", va="center", fontsize=8)
        ax.text(f + 0.01, yi - 0.2, f"{f:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("test-set score")
    ax.set_title("Test-set comparison across systems")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "06_model_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def _save_reports(config, payload, results) -> None:
    (config.reports_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    lines: list[str] = [
        "# Benchmark summary (generated by scripts/run_study.py)",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "**Scope:** classification benchmark on Financial PhraseBank "
        "(Malo et al. 2014) - label-level metrics only, no trading claims.",
        "",
        "## Data",
        "",
        f"- Corpus: `{payload['dataset']['hf_repo']}` :: "
        f"{payload['dataset']['config_name']} "
        f"({payload['dataset']['n_rows_before_dedup']} raw labeled lines -> "
        f"{payload['dataset']['n_rows_after_dedup']} unique sentences after "
        "exact-text dedup)",
        f"- Class counts: {payload['dataset']['class_counts']}",
        f"- Class proportions: {payload['dataset']['class_proportions']}",
        f"- Splits: {payload['dataset']['split_sizes']} (stratified, seed=42)",
        "",
        "## Test-set results",
        "",
        results.to_markdown(index=False),
        "",
        "## Per-class F1 (test)",
        "",
        "| model | negative | neutral | positive |",
        "| --- | --- | --- | --- |",
    ]
    for r in payload["results"]:
        pc = r["per_class"]
        lines.append(
            f"| {r['model']} | {pc['negative']['f1']:.3f} | "
            f"{pc['neutral']['f1']:.3f} | {pc['positive']['f1']:.3f} |"
        )
    m = payload["mcnemar"]
    lines += [
        "",
        "## McNemar test (two best systems)",
        "",
        f"- Systems: {m['system_a']} vs {m['system_b']}",
        f"- b (A right / B wrong): {m['b_a_right_b_wrong']}; "
        f"c (A wrong / B right): {m['c_a_wrong_b_right']}",
        f"- Exact two-sided p = {m['p_value']:.3e} -> {m['verdict']}",
        "",
        "## Hyperparameter tuning (validation macro-F1)",
        "",
    ]
    for key, block in payload["tuning"].items():
        recs = ", ".join(
            f"C={r['C']}: {r['val_macro_f1']:.4f}" for r in block["grid_records"]
        )
        lines.append(f"- {key}: best C={block['best_C']} ({recs})")
    lines += ["", "## 5-fold stratified CV on train", ""]
    for key, cv in payload["cv_train"].items():
        lines.append(
            f"- {key}: macro-F1 {cv['mean_macro_f1']:.4f} +/- {cv['std_macro_f1']:.4f}"
        )
    lines += [
        "",
        "## FinBERT runtime",
        "",
        f"- Coverage: {payload['finbert']['coverage']} "
        f"(n={payload['finbert']['n_evaluated']})",
        f"- Throughput: {payload['finbert']['seconds_per_1000_sentences']} s / 1000 "
        f"sentences; {payload['finbert']['total_seconds']} s total incl. model load",
        f"- Caveat: {payload['finbert']['caveat']}",
        "",
        "## Error analysis examples",
        "",
    ]
    for e in payload["error_analysis_examples"]:
        lines.append(
            f"- [{e['true']} -> {e['predicted']}] \"{e['sentence']}\" "
            f"(VADER: {e['vader_pred']}, FinBERT: {e['finbert_pred']})"
        )
    lines += [
        "",
        f"Total wall time: {payload['total_wall_seconds']} s",
        "",
    ]
    (config.reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("reports written to %s", config.reports_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_pipeline()
    print("done")
