# financial-nlp-sentiment

![CI](https://github.com/aydinmonavvari/financial-nlp-sentiment/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 1 · Short description

Benchmarking financial sentiment classification on the *Financial PhraseBank* corpus
(Malo et al. 2014): a general-domain lexicon (VADER), TF-IDF + logistic regression,
frozen sentence embeddings (all-MiniLM-L6-v2) + logistic regression, and a
domain-specific pretrained model (FinBERT, inference only) — evaluated with stratified
splits, macro-F1, per-class diagnostics and a McNemar comparison.

*Educational research project. This is a classification benchmark: it makes no trading,
profitability or market-prediction claims of any kind.*

## 2 · Research question

> How far does each rung of the sentiment-classification ladder — a general-domain
> lexicon, bag-of-words features, frozen sentence embeddings, and a domain-specific
> pretrained transformer used inference-only — get on labeling financial-news
> sentences with negative/neutral/positive sentiment, and where exactly does each
> approach break down?

## 3 · Motivation

Financial sentiment analysis is a common component in research and industry NLP
stacks, but its difficulty is routinely underestimated. Finance prose is terse,
jargon-heavy, and dominated by the *neutral* class: "Operating profit was EUR 4.1
million" is a factual statement, not an opinion. A lexicon tuned on social media
(VADER) has no finance vocabulary; bag-of-words models see "profit" and "loss" but
miss compositional cues; frozen embeddings encode meaning without a task head; and a
domain-pretrained transformer (FinBERT) saw financial sentiment during its own
pretraining. This project measures each rung on the same splits with the same metrics,
and reports per-class behavior rather than a single accuracy number — because with
60% neutral sentences, accuracy flatters every model.

## 4 · Why this matters

- **For finance NLP:** sentiment labels feed downstream research (event studies,
  text-based indices); knowing whether a given method classifies sentences *at all*
  — before anyone wires it into a pipeline — is the prerequisite question.
- **For machine learning:** this is a controlled case study in class imbalance
  (macro-F1 vs accuracy), evaluation of lexicon → feature → pretrained-model
  baselines, and paired significance testing (exact McNemar) between two systems.
- **For evaluation hygiene:** the domain model's possible training overlap with the
  benchmark corpus is *stated, not hidden* — an example of honest reporting when a
  clean holdout comparison is not achievable with CPU-only resources.

## 5 · Methodology

1. **Data** — Financial PhraseBank (`Sentences_50Agree.txt`, the configuration where
   ≥50% of annotators agree), fetched from the Hugging Face mirror
   `takala/financial_phrasebank`, cached under `data/raw/` (never committed).
2. **Deduplication** — 4,846 raw labeled lines → 4,838 unique sentences (8 exact
   duplicate texts dropped before splitting, so no sentence can leak across splits).
3. **Splits** — stratified 70/15/15 train/validation/test (seed 42). Validation is
   used only for hyperparameter selection (logistic-regression C); test is touched once.
4. **Classical models** — 5-fold stratified CV on the training split at the selected C.
5. **VADER** — compound-score thresholds fixed a priori (≥ 0.35 → positive, ≤ −0.15 →
   negative, else neutral); never tuned on any split.
6. **FinBERT** — `ProsusAI/finbert` inference only (no fine-tuning; CPU budget),
   test split, batch 64, max_length 64.
7. **Evaluation** — accuracy, macro-F1, per-class precision/recall/F1, confusion
   matrices, multiclass Brier, exact McNemar between the two best systems, error
   analysis on the best trainable system.

## 6 · Dataset

| Property | Value |
| --- | --- |
| Source | Financial PhraseBank (Malo et al. 2014), via HF `takala/financial_phrasebank` |
| Configuration | `sentences_50agree` (≥50% annotator agreement) |
| Raw labeled lines | 4,846 → **4,838 unique sentences** after exact-text deduplication |
| Classes | neutral 2,872 (59.4%) · positive 1,362 (28.1%) · negative 604 (12.5%) |
| Language | English financial news sentences (Finnish/Nordic companies, 2003–2013) |
| Splits (actual) | 3,386 train / 726 val / 726 test (stratified, seed 42) |
| Split label proportions | identical to 0.1 pp across all three splits (see `reports/metrics.json`) |

## 7 · Data sources

The corpus was built and labeled by Malo et al. (2014) and is distributed as the
`FinancialPhraseBank-v1.0` archive (license file included in the archive; research
use — cite Malo et al. 2014). This repository accesses the Hugging Face Hub mirror
(`takala/financial_phrasebank`, file `data/FinancialPhraseBank-v1.0.zip`), caches it
locally, and never redistributes or commits it. Model weights (MiniLM ≈ 90 MB,
FinBERT ≈ 440 MB) are likewise downloaded once and never committed.

**Text encoding (verified byte-level).** The archive's member files are ISO-8859-1:
a strict UTF-8 decode fails, and the ~160 non-ASCII bytes are valid latin-1 accents
(ñ, à, ä, £, …). The loader therefore decodes `iso-8859-1` and **never edits corpus
text**. Note that the upstream archive itself ships with pre-corrupted accented
characters — e.g. the byte sequence `0x2B 0xC2` that latin-1-renders as `Bj+Ârn`
where `Björn` was intended, part of ~134 `+<accent>` artifacts in
`Sentences_50Agree.txt` (present identically in all four member files of the
official v1.0 distribution, alongside its `__MACOSX` metadata). These upstream
damages are preserved verbatim rather than "repaired": no decoding recovers the
lost characters, and rewriting corpus text (or switching to a UTF-8
`errors="replace"` decode, which would additionally destroy the 151 *legitimate*
accented characters) would fabricate data. The one affected sentence appears in the
error-analysis examples exactly as distributed.

## 8 · Architecture

```
HF Hub (takala/financial_phrasebank) ──► cache data/raw/ ──► parse sentence@label
                                                                │
                                     dedup (4846 → 4838) ──► stratified 70/15/15
                                                                │
        majority ──┐                                            │
        VADER (fixed thresholds, a priori) ──┐                    │
        TF-IDF 1-2g + LogReg (C tuned on val, 5-fold CV on train) │
        frozen MiniLM embeddings + LogReg (C tuned on val, CV)    │
        FinBERT inference (test only, batch 64)                   │
                └────────────────┬─────────────────────────────┘
                     single evaluation on test (n = 726)
                                 │
     macro-F1 · per-class P/R/F1 · confusion · Brier · exact McNemar
     error analysis (6 quoted sentences) · 6 figures
```

## 9 · Experimental design

**Why stratified (label-random) splitting is the right design here:** the corpus is an
unordered pool of standalone sentences with **no timestamps** — there is no temporal
ordering to preserve, so a time-based split is not applicable. Stratification keeps the
59/28/12 class mix identical across splits, which is exactly the comparison this
benchmark needs. This is *not* a market-prediction setting and no such claim is made.

**Metric choice under imbalance:** with 59.4% neutral sentences, the majority
baseline scores 59.4% accuracy with zero informational value. Macro-F1 weights the
three classes equally and per-class F1 exposes *where* systems fail (the negative
class has only ~91 test sentences — a 1-sentence error moves its F1 by ~0.5 points).

**A priori decisions, not post hoc:** VADER thresholds (0.35 / −0.15) were fixed
before any PhraseBank output was inspected; the C grid (0.25–8.0) and all seeds were
fixed in `config.py` before the study ran. Validation split is the *only* data used
for model selection.

**Significance:** the two best systems are compared with an exact two-sided McNemar
test on paired test-set correctness (`statsmodels.stats.contingency_tables.mcnemar`,
exact binomial on the b/c off-diagonal counts).

## 10 · Models

| # | System | Configuration |
| --- | --- | --- |
| 1 | Majority | always predicts the most frequent training class (neutral) |
| 2 | VADER | `vaderSentiment` compound score; ≥ 0.35 → positive, ≤ −0.15 → negative, else neutral (fixed a priori) |
| 3 | TF-IDF + LogReg | word 1–2 grams, sublinear TF, `min_df=2`; multinomial logistic regression, C tuned on val → **C = 4.0** |
| 4 | Frozen embeddings + LogReg | `sentence-transformers/all-MiniLM-L6-v2` (384-d, frozen, CPU) + logistic regression, C tuned on val → **C = 4.0** |
| 5 | FinBERT | `ProsusAI/finbert`, **inference only** (no fine-tuning), batch 64, max_length 64, test split |

All seeds pinned to 42. Preprocessing is deliberately minimal (whitespace
normalization only; case preserved — VADER is case-sensitive by design, TF-IDF
lowercases internally).

## 11 · Evaluation metrics

- **Discrimination:** accuracy and **macro-F1** (primary), per-class
  precision/recall/F1/support in canonical order negative/neutral/positive.
- **Confusion structure:** 3×3 confusion matrices (rows = true) for the two best systems.
- **Calibration note:** multiclass Brier score (mean over samples of Σₖ(pₖ−yₖ)², range
  [0, 2]) and mean maximum-class confidence for the three probabilistic systems.
- **Significance:** exact two-sided McNemar test between the two best systems.
- **Error analysis:** 6 real misclassified test sentences (best trainable system),
  cross-referenced against VADER and FinBERT predictions.

## 12 · Results

Actual outputs of the committed run (`reports/metrics.json`, generated by
`scripts/run_study.py`; the run reproduces deterministically — two runs gave
identical metrics).

**Test set, n = 726 (stratified; supports: negative 91, neutral 431, positive 204):**

| System | Accuracy | Macro-F1 | Neg F1 | Neu F1 | Pos F1 |
| --- | --- | --- | --- | --- | --- |
| Majority baseline | 0.5937 | 0.2483 | 0.000 | 0.745 | 0.000 |
| VADER (fixed thresholds) | 0.5868 | 0.4995 | 0.347 | 0.688 | 0.463 |
| TF-IDF + LogReg (C=4.0) | 0.7562 | 0.6989 | 0.671 | 0.827 | 0.599 |
| MiniLM + LogReg (C=4.0) | 0.7769 | 0.7334 | 0.710 | 0.840 | 0.650 |
| FinBERT (inference only) | **0.8705** | **0.8647** | 0.864 | 0.887 | 0.843 |

**Calibration note (multiclass Brier, lower is better; range [0, 2]):** TF-IDF 0.339 ·
MiniLM 0.331 · FinBERT 0.188. Mean maximum-class confidence: 0.761 / 0.795 / 0.872 —
all three systems are overconfident relative to their accuracy, FinBERT least so.

**Hyperparameter tuning (validation macro-F1):** TF-IDF best C=4.0 (0.7172; C=8.0
gives 0.7162), MiniLM+LR best C=4.0 (0.7431; C=8.0 gives 0.7269). 5-fold stratified CV
on the training split at the selected C: TF-IDF 0.6768 ± 0.0213, MiniLM+LR
0.7276 ± 0.0140 — consistent with the test-set ordering of the two classical systems.

**McNemar test (FinBERT vs MiniLM+LR, paired test correctness):** b = 120
(FinBERT right / MiniLM wrong), c = 52 (FinBERT wrong / MiniLM right); exact two-sided
**p = 2.285e-07 → significant at α = 0.05**. The domain model's advantage is not a
fluke of this particular test split.

**Class-imbalance reading:** every system's *best* class is neutral (largest support)
and the negative class is the least stable despite being the most distinctive
semantically. FinBERT is the only system whose minority-class F1 (neg 0.864, pos 0.843)
comes close to its neutral F1 (0.887); for TF-IDF the positive-class recall is only
0.505 — half of all positive sentences are mislabeled, mostly as neutral.

Figures: [`01_class_distribution.png`](figures/01_class_distribution.png) ·
[`02_confusion_matrices.png`](figures/02_confusion_matrices.png) (the two best
systems) · [`03_per_class_f1.png`](figures/03_per_class_f1.png) (imbalance view) ·
[`04_embedding_pca.png`](figures/04_embedding_pca.png) (frozen-embedding PCA by true
label — neutral dominates the space) ·
[`05_error_analysis.png`](figures/05_error_analysis.png) ·
[`06_model_comparison.png`](figures/06_model_comparison.png).

## 13 · Interpretation

1. **There is a clean monotone ladder.** Majority (mF1 0.248) → VADER (0.500) →
   TF-IDF (0.699) → MiniLM (0.733) → FinBERT (0.865). Each design improvement buys a
   real, measurable gain; no rung is redundant.
2. **VADER fails exactly as the literature predicts.** Macro-F1 0.500 looks
   respectable until read per class: 0.347 negative F1 at 0.47 precision. Its
   general-domain lexicon scores terse finance prose ("The share was traded at…")
   unpredictably, and its neutral band (−0.15, 0.35) swallows informative sentences —
   60.7% of test predictions land in neutral (441 of 726). A lexicon is not a baseline to skip; it
   is a baseline to beat, decisively.
3. **Frozen embeddings beat bag-of-words modestly, and mostly on minority classes.**
   MiniLM+LR gains +0.035 macro-F1 over TF-IDF overall but +0.039 negative F1 and
   +0.051 positive F1, at the cost of negative precision (0.813 → 0.707) as it trades
   majority-leaning conservatism for minority-class recall. Compositional meaning
   (negation scope, "from X to Y" patterns) helps precisely where the minority classes
   live. The 5-fold CV gap (0.728 vs 0.677) confirms the ordering is not split luck.
4. **The domain model is in a different league — with an asterisk.** FinBERT's +0.13
   macro-F1 over the best trainable system is statistically significant (McNemar
   p = 2.3e-07), and its error profile differs qualitatively: recall ≥ 0.83 on every
   class. But FinBERT was pretrained/fine-tuned by its authors on financial text
   including PhraseBank-style sentences, so this is a *domain-model reference point*,
   not a clean holdout comparison (see §14). The honest conclusion: a general
   sentence encoder plus a logistic regression gets you ~0.73 macro-F1 for 90 MB and
   minutes of CPU; the domain model gets ~0.86 and the difference is exactly the value
   of domain pretraining.
5. **The neutral class is the background rate.** All systems' neutral F1 clusters at
   0.69–0.89 while minority-class F1 spreads 0.00–0.86. Confusion flows overwhelmingly
   minority → neutral for the classical systems (`02_confusion_matrices.png`):
   TF-IDF mislabels 98 of 204 positive test sentences as neutral (positive recall 0.505).

## 14 · Limitations

- **Single corpus, sentence level.** Financial PhraseBank sentences are short
  news-wire excerpts from 2003–2013, labeled by crowdsourced annotators with ≥50%
  agreement (config `sentences_50agree`); annotator disagreement noise is baked in.
  Results do not transfer to documents, social media, or other domains.
- **FinBERT overlap caveat (important).** ProsusAI/finbert's training corpus includes
  financial text of exactly this style, and its authors report using
  Financial-PhraseBank-like data. Its scores are therefore a *domain-model reference*,
  not a clean holdout result — the FinBERT-vs-others gap is an upper bound on what
  domain pretraining contributes, and FinBERT's own numbers are inflated relative to
  an honest unseen-corpus estimate. We report it as measured, with this caveat, and do
  not fine-tune anything (CPU budget; fine-tuning comparison is future work).
- **No fine-tuning.** All trainable systems use logistic-regression heads;
  fine-tuning MiniLM or BERT on the training split would raise every learned baseline
  and is the natural next experiment.
- **No trading, profitability or market-prediction claims.** This is a classification
  benchmark; it does not simulate strategies, use price data, or evaluate economic value.
- **One seed, one split.** All results use seed 42; McNemar covers the two best
  systems only. Bootstrap confidence intervals and multi-seed variance are future work.
- **VADER thresholds are a design choice.** Fixed a priori (0.35/−0.15), not tuned;
  alternative mappings could improve VADER's numbers, but tuning a "baseline" on the
  validation split would blur the study's a-priori design.

## 15 · Reproducibility

```bash
# 1) environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]          # offline test suite + lint
pip install -e .[nlp]          # only if you want to run the full study (torch etc.)

# 2) full study (~2 min on CPU after first download; seed = 42, deterministic)
python scripts/run_study.py    # first run downloads corpus + models, then caches

# 3) verify: lint + 30 offline tests
ruff check .
pytest -q
```

The study is deterministic: same seed, same splits, same metrics (two independent
runs gave identical tables). Total wall time ≈ 149 s on CPU (FinBERT test inference
82 s, MiniLM embedding of the full corpus 42 s; exact per-run timings are recorded
in `reports/metrics.json`). Heavy model tests are guarded with
`pytest.importorskip`, so the 28 light tests pass with base deps only (CI does this).

## 16 · Installation

```bash
git clone https://github.com/aydinmonavvari/financial-nlp-sentiment.git
cd financial-nlp-sentiment
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]      # base deps + pytest + ruff (no torch needed for tests)
```

Base dependencies: numpy, pandas, scikit-learn, matplotlib, scipy, statsmodels,
vaderSentiment, huggingface_hub, pyarrow (parquet writer for the split-assignment
audit artifact). The `nlp` extra adds torch, transformers and
sentence-transformers (only needed for the embedding/FinBERT parts of the study).

## 17 · Usage

```bash
python scripts/run_study.py                      # full benchmark, all outputs
python scripts/run_study.py --no-figures         # metrics + reports only
python scripts/run_study.py --seed 7             # different split (for robustness)
pytest -q                                        # 30 offline tests (28 run on CI)
```

As a library:

```python
from financial_nlp_sentiment.config import DEFAULT_CONFIG
from financial_nlp_sentiment.data import load_phrasebank, make_splits
from financial_nlp_sentiment.text import vader_predict
from financial_nlp_sentiment.evaluation import macro_f1

frame = load_phrasebank(DEFAULT_CONFIG)          # downloads once, then cached
splits = make_splits(frame, DEFAULT_CONFIG)
test = frame.iloc[splits.test]
_, preds = vader_predict(test["sentence"].tolist())
print(macro_f1(test["label"].tolist(), preds))   # 0.4995 for the test split
```

## 18 · Example

Actual misclassified test sentences (best trainable system: MiniLM + LogReg; full
table in `reports/summary.md` and `figures/05_error_analysis.png`):

- **[positive → neutral]** *"The airline has ordered nine Airbus A350-900 aircraft
  with deliveries from 2011 …"* — a large order is positive news for the supplier;
  the sentence's surface form is a neutral fact. World knowledge, not wording, decides.
- **[neutral → negative]** *"As a consequence of the merger the number of
  administration personnel … will be reduced by six."* — "reduced" is lexically
  negative but annotators read it as neutral factual reporting (n = 6, small scale).
  The classic neutral-but-informative trap.
- **[negative → positive]** *"The result will also be burdened by increased fixed
  costs … and restructuring costs in Japan."* — "burdened" should dominate, but
  "increased" and "costs" pull the embedding toward neutral/positive territory;
  VADER gets this one right (negative).
- **[positive → negative]** *"The company moved to an operating profit of EUR10.9 m
  versus an operating loss of EUR15.3 m…"* — "loss" appears literally; the flip from
  loss to profit requires reading the "versus" structure. A compositional failure of
  every system except FinBERT (predicted positive).

The pattern: classical systems stumble on composition (negation scope, before/after
contrast, "reduced by six" as neutral fact) — exactly the phenomena that domain
pretraining encodes.

## 19 · Project structure

```
financial-nlp-sentiment/
├── README.md
├── LICENSE · CITATION.cff · pyproject.toml · requirements.txt
├── .python-version · .gitignore
├── src/financial_nlp_sentiment/
│   ├── config.py        # seed, splits, VADER thresholds, C grid, paths
│   ├── data.py          # HF fetch + cache, sentence@label parser, dedup, stratified splits
│   ├── text.py          # minimal preprocessing + VADER lexicon baseline
│   ├── models.py        # majority, TfidfLogReg, EmbeddingLogReg, FinBERT, tuning + CV
│   ├── evaluation.py    # macro-F1, per-class, confusion, Brier, exact McNemar, errors
│   └── pipeline.py      # orchestration, figures, reports
├── tests/               # 30 offline tests (heavy imports guarded for CI)
├── scripts/run_study.py # CLI: --seed, --test-size, --val-size, --finbert-batch-size
├── data/raw · data/processed  # git-ignored caches (.gitkeep tracked)
├── reports/             # metrics.json + summary.md (generated, committed)
├── figures/             # 6 generated figures (committed)
├── docs/research_report.md
└── .github/workflows/ci.yml
```

## 20 · Future work

- Fine-tune MiniLM/DistilBERT on the training split (needs GPU or patience) to close
  the frozen-vs-finetuned gap.
- FinBERT on the full corpus with probability calibration (temperature scaling on val).
- Bootstrap confidence intervals over the test set; repeat across 5 seeds.
- Loughran–McDonald domain lexicon as a finance-specific lexicon baseline (gated
  vocabulary — not redistributed here; cited as related work).
- Extend to the higher-agreement configurations (66/75/100agree) to measure label-noise
  sensitivity.

## 21 · Citation

If you use this work, please cite (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{monavvari2026financialnlpsentiment,
  author  = {Monavvari, Aydin},
  title   = {financial-nlp-sentiment: benchmarking financial sentiment classification on the Financial PhraseBank corpus},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/aydinmonavvari/financial-nlp-sentiment}
}
```

Dataset citation: Malo, P., Sinha, A., Korhonen, P., Pyysalo, S., & Takala, P. (2014).
Good debt or bad debt: Detecting semantic orientations in economic texts.
*Journal of the Association for Information Science and Technology*, 65(4), 782–796.

## 22 · License

MIT — see [`LICENSE`](LICENSE). The Financial PhraseBank corpus and all model weights
remain under their own licenses and are not redistributed by this repository.

## 23 · Acknowledgments

Corpus by Malo et al. (2014) (Aalto University / University of Helsinki), HF mirror by
Pekka Takala. Models: VADER (Hutto & Gilbert), all-MiniLM-L6-v2 (Reimers & Gurevych
line of work; Sentence-Transformers), FinBERT (Araci 2019 / Prosus AI). Built with
pandas, scikit-learn, vaderSentiment, sentence-transformers, transformers and matplotlib.
