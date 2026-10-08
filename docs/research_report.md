# Research Report — Financial NLP Sentiment

**Benchmarking sentiment classification on the Financial PhraseBank corpus: lexicon, bag-of-words, frozen embeddings, and a domain-pretrained reference model**

*Author: Aydin Monavvari — educational research project (classification benchmark only; no trading, profitability or market-prediction claims).*
*Run date: 2026-10-08. All numbers are actual outputs of the committed pipeline (`reports/metrics.json`).*

---

## Abstract

This study benchmarks five approaches to three-class financial sentiment classification (negative/neutral/positive) on the Financial PhraseBank corpus, configuration `sentences_50agree` (Malo et al. 2014; 4,846 labeled lines → 4,838 unique sentences after exact-text deduplication; 59.4% neutral, 28.1% positive, 12.5% negative). The systems span the standard design ladder: (1) a majority-class baseline, (2) VADER, a general-domain lexicon (Hutto & Gilbert 2014) mapped to three classes via a priori compound-score thresholds, (3) word 1–2 gram TF-IDF features with multinomial logistic regression (Pedregosa et al. 2011), (4) frozen all-MiniLM-L6-v2 sentence embeddings (Reimers & Gurevych 2019) with logistic regression, and (5) FinBERT (Araci 2019), a BERT-family model (Devlin et al. 2019) pretrained for financial text, used inference-only. With a stratified 70/15/15 split (seed 42), test-set macro-F1 rises monotonically across the ladder: 0.248 (majority), 0.500 (VADER), 0.699 (TF-IDF+LR, C=4.0), 0.733 (MiniLM+LR, C=4.0), 0.865 (FinBERT). The McNemar exact test between the two best systems (FinBERT vs MiniLM+LR) is significant (b=120, c=52, p=2.3e-07). FinBERT's advantage must be read cautiously: its training corpus includes PhraseBank-style financial text, so its scores are treated as a domain-model reference rather than a clean holdout result. Per-class analysis shows the imbalance structure (neutral support 431 vs negative 91 on test) dominates the classical systems' error patterns: minority-class sentences flow into neutral, and compositional cues (loss-to-profit contrasts, negation scope, neutral-but-informative fact reporting) account for most of the quoted errors.

## Introduction

Sentiment analysis of financial text is widely used as an input to research and monitoring pipelines, yet the difficulty of the task is easy to misjudge. Financial news prose is terse and factual; the majority of sentences carry *no* sentiment in the annotator's judgment even when they contain financially informative content ("Operating profit decreased to EUR 4.1 million from EUR 6.2 million" is, per the PhraseBank annotation scheme, neutral-negative leaning but often labeled neutral when annotators disagree). A lexicon built for social media does not speak this dialect; bag-of-words models capture lexical polarity but not composition; frozen sentence embeddings encode meaning but have no task-specific head; and a transformer pretrained on financial text has already seen the task's distribution.

This project measures each rung of that ladder on identical splits and metrics. The design deliberately isolates *classification quality* as the object of study. Nothing in this repository simulates trading, touches price data, or claims that better sentence classification implies economic value — the gap between label accuracy and decision value is a separate research problem.

## Research Question

> How do four qualitatively different sentiment-classification designs — a general-domain lexicon, TF-IDF bag-of-words features, frozen sentence embeddings, and an inference-only domain-pretrained transformer — compare on the Financial PhraseBank three-class task, and what do per-class results and quoted errors reveal about where each design fails?

## Related Work

- **Malo, P., Sinha, A., Korhonen, P., Pyysalo, S., & Takala, P. (2014).** "Good debt or bad debt: Detecting semantic orientations in economic texts." *Journal of the Association for Information Science and Technology*, 65(4), 782–796. Source of the corpus and of the annotation-agreement configurations (50/66/75/100 agree).
- **Hutto, C. J., & Gilbert, E. (2014).** "VADER: A parsimonious rule-based model for sentiment analysis of social media text." *ICWSM '14*. The general-domain lexicon baseline; used unchanged with fixed thresholds.
- **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).** "BERT: Pre-training of deep bidirectional transformers for language understanding." *NAACL-HLT 2019*. Architecture family underlying the domain model.
- **Araci, D. (2019).** "FinBERT: Financial sentiment analysis with pre-trained language models." *arXiv:1908.10063*. The domain-pretrained model used here inference-only (`ProsusAI/finbert`); its authors fine-tuned BERT on financial text, which motivates the overlap caveat (§Limitations).
- **Reimers, N., & Gurevych, I. (2019).** "Sentence-BERT: Sentence embeddings using Siamese BERT-networks." *EMNLP 2019*. The sentence-embedding approach instantiated here by the all-MiniLM-L6-v2 checkpoint.
- **Loughran, T., & McDonald, B. (2011).** "When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks." *Journal of Finance*, 66(1), 35–65. Cited as the canonical finance-specific lexicon line of work; **not used** in this study (the dictionary is gated and is not redistributed), but it motivates why general-domain lexicons misfire on finance text.
- **Pedregosa, F., et al. (2011).** "Scikit-learn: Machine learning in Python." *JMLR*, 12, 2825–2830. TF-IDF vectorization, logistic regression, splits, CV and metrics.
- **McNemar, Q. (1947).** "Note on the sampling error of the difference between correlated proportions or percentages." *Psychometrika*, 12(2), 153–157. The paired significance test (exact binomial variant via statsmodels).

## Data

The Financial PhraseBank (Malo et al. 2014) contains 4,846 sentences extracted from financial news (LexisNexis) about Finnish/Nordic listed companies (2003–2013), each labeled by 5–8 annotators and again by finance researchers; the `50agree` configuration retains sentences where at least 50% of annotators agree. The archive is fetched from the Hugging Face Hub mirror `takala/financial_phrasebank` (file `data/FinancialPhraseBank-v1.0.zip`), cached under `data/raw/` and never committed. The archive ships with its own license file; research use with citation is the intended purpose.

After parsing `sentence@label` lines (ISO-8859-1 decoding) and dropping 8 exact duplicate sentence texts (first occurrence kept) to prevent cross-split leakage, the study corpus is **4,838 unique sentences**: neutral 2,872 (59.36%), positive 1,362 (28.15%), negative 604 (12.48%). Neutral is the majority class — the defining property that makes accuracy misleading and stratification, macro-F1 and per-class reporting necessary.

Splits are stratified 70/15/15 (3,386 / 726 / 726, seed 42); label proportions match across splits to within 0.1 percentage points (exact values in `reports/metrics.json`).

## Methodology

**Preprocessing policy.** Whitespace normalization only; case preserved (VADER's lexicon is case-sensitive by design; TF-IDF lowercases internally as an explicit, documented model-side choice). No stemming, stopword removal or punctuation stripping — negation and "%" tokens are signal, not noise, for this task.

**Systems.**
1. *Majority baseline*: predicts the training majority class (neutral) for every instance; ties broken deterministically.
2. *VADER*: compound score thresholds fixed a priori — ≥ 0.35 → positive, ≤ −0.15 → negative, otherwise neutral. The asymmetric neutral band is a documented design decision (mildly positive finance prose inflates compound scores); thresholds were set before inspecting any PhraseBank output and are not tuned on any split.
3. *TF-IDF + LogReg*: word 1–2 grams, sublinear TF, `min_df=2`, multinomial logistic regression; C selected on the validation split from {0.25, 0.5, 1, 2, 4, 8} by macro-F1 → C = 4.0.
4. *Frozen embeddings + LogReg*: all-MiniLM-L6-v2 (384-d, CPU, frozen — no fine-tuning of the encoder), logistic regression head; same C grid → C = 4.0.
5. *FinBERT*: `ProsusAI/finbert`, softmax probabilities remapped to the canonical label order; **inference only** — no parameters are updated, no training data is touched. Evaluated on the test split (n = 726), batch 64, max_length 64; ~54 s per 1,000 sentences on CPU, 41 s total for the test split including model load.

**Model selection and validation.** The validation split is used only for C selection; 5-fold stratified cross-validation on the *training* split at the selected C is reported as a stability check for the two classical models (TF-IDF 0.677 ± 0.021; MiniLM+LR 0.728 ± 0.014). The test split is evaluated exactly once per system.

**Evaluation.** Accuracy, macro-F1, per-class precision/recall/F1/support, 3×3 confusion matrices, multiclass Brier score (Σₖ(pₖ−yₖ)² averaged over samples, range [0, 2]) and mean maximum-class confidence as a calibration sanity check, exact two-sided McNemar between the two best systems, and a 6-sentence error analysis sampled round-robin across confusion pairs from the best trainable system.

## Experimental Design

**Why a stratified label-random split (and not a temporal one).** The corpus is an unordered pool of standalone sentences with no publication timestamps attached to individual rows; there is no temporal ordering to preserve. Stratification preserves the 59/28/12 class mix in every split, which is the controlled comparison this benchmark requires. This is a document-classification setting, *not* a forecasting setting — no claim about future market behavior is made or testable with this design.

**Why macro-F1 is primary.** The majority baseline achieves 59.4% accuracy while carrying zero information about the minority classes (their F1 is 0.000). Macro-F1 averages class-wise F1 with equal weight, so a system cannot score well by ignoring a class; per-class F1 then localizes *where* failures occur. The negative class (91 test sentences) is small enough that single-sentence errors move its F1 by ~0.5 points — supports are therefore always reported alongside F1.

**A priori choices.** VADER thresholds, the C grid, fold count, seed and the FinBERT inference budget were fixed in configuration before the study ran; none were adjusted after seeing results.

**Pairwise significance.** The two best systems by test macro-F1 are compared with the exact McNemar test on paired correctness (b = one system right / other wrong; exact binomial two-sided p on the off-diagonal counts).

## Results

Test set n = 726 (supports: negative 91, neutral 431, positive 204).

| System | Accuracy | Macro-F1 | Neg F1 | Neu F1 | Pos F1 |
| --- | --- | --- | --- | --- | --- |
| Majority | 0.5937 | 0.2483 | 0.000 | 0.745 | 0.000 |
| VADER (a priori thresholds) | 0.5868 | 0.4995 | 0.347 | 0.688 | 0.463 |
| TF-IDF + LogReg (C=4.0) | 0.7562 | 0.6989 | 0.671 | 0.827 | 0.599 |
| MiniLM + LogReg (C=4.0) | 0.7769 | 0.7334 | 0.710 | 0.840 | 0.650 |
| FinBERT (inference only) | 0.8705 | 0.8647 | 0.864 | 0.887 | 0.843 |

Per-class precision/recall for the two leading systems: FinBERT neg P/R = 0.774/0.978, neu 0.952/0.831, pos 0.787/0.907; MiniLM+LR neg 0.707/0.714, neu 0.796/0.889, pos 0.758/0.569.

**Calibration note.** Multiclass Brier: TF-IDF 0.339, MiniLM 0.331, FinBERT 0.188; mean maximum-class confidence 0.761, 0.795, 0.872 respectively. All three probabilistic systems are overconfident relative to their accuracy (e.g., MiniLM+LR asserts its label with mean 0.795 while being right 0.777 of the time); no calibration procedure was applied — this is reported as a measurement, with temperature scaling / Platt calibration on the validation split as natural follow-up.

**Validation and CV.** Validation macro-F1 at the selected C: TF-IDF 0.717, MiniLM+LR 0.743 (grids in `reports/metrics.json`). 5-fold stratified CV on train: TF-IDF 0.6768 ± 0.0213, MiniLM+LR 0.7276 ± 0.0140 — the classical systems' ordering is stable across folds, not a split artifact.

**McNemar (FinBERT vs MiniLM+LR).** b = 120 (FinBERT correct, MiniLM wrong), c = 52 (FinBERT wrong, MiniLM correct); exact two-sided **p = 2.285e-07** — significant at α = 0.05.

**Error analysis** (best trainable system: MiniLM+LR; six real test sentences, cross-referenced with VADER and FinBERT):

| True → Pred | Sentence (abridged) | VADER | FinBERT | Failure mode |
| --- | --- | --- | --- | --- |
| pos → neu | "The airline has ordered nine Airbus A350-900 aircraft…" | neutral | positive | positivity requires world knowledge (large order = good for supplier) |
| neu → pos | "…Sampo Bank's acquisition… was the largest cash deal in Finland's economic history." | neutral | positive | superlative phrasing reads as positive; annotators label the fact neutral |
| neg → neu | "Those uncertainties cloud the long-term outlook." | negative | negative | hedged, jargon-light negativity missed by the learned head |
| neu → neg | "…the number of administration personnel … will be reduced by six." | neutral | neutral | "reduced" is lexically negative; annotators treat small-scale fact as neutral (neutral-but-informative) |
| pos → neg | "…moved to an operating profit of EUR10.9 m versus an operating loss…" | neutral | positive | "loss" appears literally; the loss→profit contrast requires compositional reading |
| neg → pos | "The result will also be burdened by increased fixed costs…" | negative | negative | "burdened" should dominate; "increased/costs" pull the embedding positive |

Figures: `figures/01_class_distribution.png`, `02_confusion_matrices.png`, `03_per_class_f1.png`, `04_embedding_pca.png`, `05_error_analysis.png`, `06_model_comparison.png`.

## Discussion

1. **The ladder is real and each rung is justified.** Every design improvement produced a measurable macro-F1 gain: +0.25 (lexicon over majority), +0.20 (supervised bag-of-words over lexicon), +0.035 (frozen embeddings over bag-of-words), +0.13 (domain model over frozen embeddings). The classical systems' ordering is confirmed by 5-fold CV on the training split, so it is not an artifact of one split.
2. **The lexicon's failure is structural, not a tuning artifact.** VADER has no representation of finance-specific constructs (a "profit warning", a "downgrade") and its neutral band is miscalibrated for terse factual prose; 60.7% of its test predictions are neutral and its negative-class F1 is 0.347. We deliberately did not tune its thresholds on PhraseBank: a baseline tuned on the evaluation corpus is no longer an independent baseline. Loughran & McDonald's (2011) finance dictionary is the standard remedy and is cited as such (not used here — gated distribution).
3. **Frozen embeddings buy minority-class recall.** MiniLM+LR's gains over TF-IDF concentrate on negative (+0.039 F1) and positive (+0.051 F1) classes, at the cost of negative precision (0.813 → 0.707). Sentence-level semantics help exactly where the class is semantically distinctive but lexically diffuse; the PCA view of the frozen embedding space (`04_embedding_pca.png`) shows the neutral class occupying most of the space with minority classes interleaved — a logistic head on frozen features can only go so far, which is precisely what fine-tuning would address.
4. **The domain-model gap is large, significant — and should be quoted with its caveat.** FinBERT's macro-F1 advantage over the best trainable system (+0.131) is significant under the exact McNemar test (p = 2.3e-07), and its error profile is qualitatively different (minority-class recall ≥ 0.90). However, FinBERT's own training corpus includes PhraseBank-style financial sentences (Araci 2019), so the model has effectively *seen the task's distribution*. Its 0.865 is best read as an upper-bound reference for domain pretraining, not as a clean holdout score. The honest statement this study can make: *under an inference-only budget, the domain model dominates; how much of that dominance would survive a strict provenance firewall cannot be determined here.*
5. **Class imbalance shapes everything.** The classical systems' errors flow overwhelmingly minority → neutral (TF-IDF: 98 of 204 positive sentences predicted neutral). With 59.4% neutral, any under-confident system regresses toward neutral, and macro-F1 punishes exactly that. The supports (91/431/204) belong next to every per-class number in this report.

## Limitations

- **FinBERT train/test overlap (primary caveat).** The domain model's pretraining/fine-tuning corpus may overlap with Financial PhraseBank-style data; its scores are a domain-model reference, not a clean holdout result. No provenance firewall can be established post hoc at inference-only budget.
- **No fine-tuning.** CPU budget restricts learned systems to frozen features + linear heads. Fine-tuning MiniLM/BERT on the 3,386-sentence training split would likely lift all trainable systems and sharpen the comparison.
- **Single corpus, sentence level, one language domain.** Results apply to short English financial news sentences (2003–2013, Finnish/Nordic companies); they do not transfer to documents, social media, earnings-call transcripts, or other markets.
- **Label noise.** `50agree` is the *lowest*-agreement configuration; annotator disagreement is baked into the gold labels, deflating every system's ceiling (especially on neutral/negative boundaries).
- **One seed, one split, single test evaluation.** The McNemar test covers only the top pair; bootstrap CIs and multi-seed replications are future work.
- **VADER thresholds are a design choice.** Fixed a priori and not tuned; other mappings exist in the literature.
- **No economic claims.** The study measures sentence-label classification only; it does not, and cannot, speak to trading profitability or market prediction.

## Conclusion

On the Financial PhraseBank `50agree` task, sentiment-classification quality rises monotonically with design sophistication: majority 0.248 → VADER 0.500 → TF-IDF+LR 0.699 → frozen MiniLM+LR 0.733 → FinBERT 0.865 macro-F1 on a stratified 726-sentence test set, with the top-two difference significant under an exact McNemar test (p = 2.3e-07). The per-class structure is the real story: neutral dominates the corpus, classical systems bleed minority-class sentences into neutral, and the remaining errors are compositional (contrast structures, negation scope, neutral-but-informative facts). The domain model's headline advantage carries an explicitly stated overlap caveat. The study demonstrates the evaluation craft — stratification, macro-F1 with supports, a priori thresholds, CV stability checks, paired significance, quoted errors — that any credible NLP benchmark needs, and it confines its claims to exactly what was measured.

## Future Research

- Fine-tune MiniLM and a BERT-base on the training split to quantify the frozen-vs-finetuned gap on this corpus.
- Provenance-controlled evaluation: assemble a held-out corpus assembled *after* FinBERT's training cutoff or from disjoint sources, to get a clean domain-model estimate.
- Calibration (temperature scaling / vector scaling on the validation split) and selective prediction (abstention on low-confidence sentences).
- Repeat across the 66/75/100agree configurations to measure label-noise sensitivity; multi-seed and bootstrap intervals.
- Loughran–McDonald dictionary as a finance-lexicon baseline (license permitting) and the addition of a negation-aware rule system.

## References

- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of ICWSM '14*.
- Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65. *(Related work; not used — gated dictionary.)*
- Malo, P., Sinha, A., Korhonen, P., Pyysalo, S., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP 2019*.
