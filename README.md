# CxER: Contextual Error Rate for ASR Evaluation

Code and data for **"The Fine-tuning Paradox: Rethinking ASR Evaluation in Safety-Critical Settings"**.

CxER is an LLM-ensemble evaluation framework that scores whether an ASR
transcription preserves *operational meaning*, rather than surface lexical
similarity. Applied to air traffic control (ATC) transcription, it exposes a
**Finetuning Paradox**: standard metrics (WER, BERTScore, SemDist) improve
substantially after fine-tuning an ASR model on ATC data, while the
concentration of *operationally critical* errors among the outputs those same
metrics rate as high-quality does not improve nearly as much — sometimes not
at all.

> **Note on naming:** the codebase uses `CCER` (Contextual Critical Error
> Rate) internally — `CER` was already taken by Character Error Rate. The
> paper uses `CxER` throughout; they are the same metric.

---

## 1. Repository map

```
cxer-release/
├── evaluate_asr.py                 # Step 2: WER / WeightedWER / BERTDist / SemDist / CxER per file
├── weighted_wer.py                 # Rule-based ATC entity tagger + Weighted WER (no LLM)
├── analysis/
│   ├── figure1_wer_scatter.py        WER-vs-CxER scatter (Fig. 1)
│   ├── figure2_metric_comparison.py  metric-improvement-gap bar chart (Fig. 2)
│   ├── table1_finetuning_paradox.py  Table 1 — the Finetuning Paradox
│   ├── table2_inter_rater_agreement.py  Table 2 — CxER validation vs. human SMEs
│   ├── table3_prompt_robustness.py   Table 3 — prompt-variant robustness
│   └── error_type_analysis.py        Table 4 / Appendix B — per-error-type agreement
├── asr/scripts/                    train_whisper.py, train_parakeet.py, asr_inference.py
├── judge/
│   ├── prompts/                      cot_v3.py (canonical), compressed.py, no_examples.py, lenient.py, base.py, cot.py
│   └── scripts/                      llm_judge.py, prompt_perturbation.py, error_type_annotation.py, llm_prompt_perturbation.py
├── data/raw/                       12 ASR transcript JSONs (2 ASR models × 2 variants × 3 test sets)
├── annotations/{llama_70b,qwen_72b,deepseek_70b}/   36 judge-output JSONs (12 files × 3 judges)
├── eval_cache/                     cached corpus metrics, SME annotations, robustness benchmark
├── prompt_robustness_outputs/      raw judge outputs for the prompt-robustness experiment
├── recover_main_annotations.py     one-time script: recovered JSON-parse failures in annotations/
├── verify_compressed_fix.py        one-time script: verifies the compressed/no_examples prompt fix
└── requirements.txt / asr_requirements.txt / nvidia_requirements.txt
```

---

## 2. Setup

Three requirement files, because vLLM (`torch>=2.9`) and NeMo (`torch==2.2`,
used for Parakeet fine-tuning) have incompatible pinned dependencies — use a
separate venv per environment.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CxER judge pipeline + Weighted WER baseline
```

| File | Use for |
|---|---|
| `requirements.txt` | CxER evaluation pipeline (vLLM judges), Weighted WER, analysis/plotting |
| `asr_requirements.txt` | Whisper fine-tuning |
| `nvidia_requirements.txt` | Parakeet fine-tuning (NeMo) |

Gated HuggingFace models (Llama-3.3, the base Whisper/Parakeet checkpoints)
need `export HF_TOKEN=your_token_here`.

---

## 3. Models

### 3.1 ASR systems under test

Two architectures, each evaluated **pretrained** (off-the-shelf checkpoint)
and **fine-tuned** ("combined", i.e. fine-tuned jointly on both training
corpora below):

| Family | Params | Pretrained checkpoint | Fine-tuned run name |
|---|---|---|---|
| Whisper | 769M (encoder-decoder Transformer) | `openai/whisper-medium` | `whisper-medium-combined` |
| Parakeet | 0.6B (FastConformer-TDT transducer) | `nvidia/parakeet-tdt-0.6b-v3` | `parakeet-tdt-0.6b-v3-combined` |

### 3.2 Fine-tuning setup

Both models are fine-tuned on the **combined training split** of ATCOSIM +
UWB-ATCC (see §4), single-GPU, seed 42 for all RNGs.

**Whisper** (`asr/scripts/train_whisper.py`, HuggingFace `Seq2SeqTrainer`):

| Hyperparameter | Value |
|---|---|
| Batch size (train / eval) | 16 / 16 |
| Gradient accumulation | 8 |
| Learning rate | 1e-5 |
| Warmup steps | 1000 |
| Epochs (max, with early stopping) | 20, patience 3 on eval WER |
| Precision | fp16 |
| Max label length | 225 tokens |
| Optimizer | AdamW (Trainer default) |
| Text normalization | `whisper_normalizer.EnglishTextNormalizer` |

**Parakeet** (`asr/scripts/train_parakeet.py`, NeMo + PyTorch Lightning):

| Hyperparameter | Value |
|---|---|
| Batch size (train / eval) | 16 / 8 |
| Gradient accumulation | 4 |
| Gradient clipping | 1.0 |
| Learning rate | 1e-5 |
| LR schedule | WarmupAnnealing, 1000 warmup steps |
| Weight decay | 0.01 |
| Optimizer betas | (0.9, 0.98) |
| Epochs (max, with early stopping) | 20, patience 5 on val WER |
| Precision | bf16-mixed |
| Optimizer | AdamW |
| Text normalization | uppercased (Parakeet convention) |
| Audio filtering | 0.1s–30.0s duration |

Both scripts select the best checkpoint by eval/val WER and push the result
to the HuggingFace Hub. `asr/scripts/asr_inference.py` then runs the frozen
pretrained and fine-tuned checkpoints over each test split to produce the
`data/raw/*_predictions.json` files that everything downstream reads.

### 3.3 LLM judges (CxER evaluators)

Three open-weight, ~70B-class instruction-tuned models, chosen to differ in
training data, alignment procedure, and reasoning behavior, run locally under
[vLLM](https://github.com/vllm-project/vllm) (no proprietary APIs, for
reproducibility and to avoid behavior drift):

| Judge | Params | HF ID |
|---|---|---|
| Llama | 70B | `meta-llama/Llama-3.3-70B-Instruct` |
| Qwen | 72B | `Qwen/Qwen2.5-72B-Instruct` |
| DeepSeek | 70B (R1-distilled) | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` |

**Inference configuration** (`judge/scripts/llm_judge.py`, the main
annotation pipeline):

| Setting | Value |
|---|---|
| Serving engine | vLLM 0.11.2 |
| Tensor parallel size | 4 GPUs per judge |
| `dtype` | float16 |
| `gpu_memory_utilization` | 0.8–0.85 |
| `max_model_len` | 8000 tokens |
| Sampling | temperature 0.0 (greedy), seed 42 |
| `max_tokens` (generation budget) | 2048 (main annotation pipeline) / 512 (prompt-robustness ablation, non-reasoning judges) / 2048 (DeepSeek, all experiments — needs room for its `<think>` block) |
| Batch size | 100–200 prompts per `llm.generate()` call |
| Reference hardware | 4× GPU per judge (paper: A100 80GB-class; this repo's fixes were verified on 8× Quadro RTX 8000 48GB, tensor-parallel 4) |

Final CxER label per sample = **majority vote** across the three judges'
independent `cot_v3` classifications.

---

## 4. Data

### 4.1 Corpora

| Corpus | Role | Approx. size | Source |
|---|---|---|---|
| ATCOSIM | fine-tuning train + in-domain test | ~10h, simulated ATC speech, non-native speakers | `Jzuluaga/atcosim_corpus` (HF) |
| UWB-ATCC | fine-tuning train + in-domain test | ~20h, real controller–pilot communications | `Jzuluaga/uwb_atcc` (HF) |
| ATCO2 | **out-of-distribution eval only**, excluded from fine-tuning | 1h test subset | `Jzuluaga/atco2_corpus_1h` (HF) |

All three are public HuggingFace datasets; official train/test splits are
used throughout — no private/institutional data.

### 4.2 Sample sizes actually evaluated (per ASR model × variant)

From `data/raw/` / `annotations/<judge>/` (identical count across
pretrained/fine-tuned since it's the same fixed test split):

| Test set | Samples |
|---|---|
| ATCO2 (OOD) | 871 |
| ATCOSIM (test) | 1,901 |
| UWB-ATCC (test) | 2,822 |
| **Total per (model, variant)** | **5,594** |

With 2 ASR models × 2 variants (pretrained/fine-tuned) × 3 judges, the main
CxER annotation pipeline produces **5,594 × 2 × 2 × 3 = 67,128** individual
judge classifications (22,376 per judge).

### 4.3 Human validation set (Table 2)

500 samples, independently annotated by **two aviation-domain doctoral
researchers** with binary `Equivalent` / `Critical_Error` labels. Stratified
by judge agreement, oversampling disagreement to concentrate the (scarce,
expensive) human-annotation budget on the operationally ambiguous cases:

| Stratum | Definition | Count |
|---|---|---|
| `agreed_equivalent` | all 3 judges unanimously said Equivalent | 150 |
| `agreed_critical` | all 3 judges unanimously said Critical_Error | 150 |
| `disagreed` | 2-vs-1 judge disagreement | 200 |
| **Total** | | **500** |

File: `eval_cache/annotation_sheet_shared.xlsx` / `.csv`.

### 4.4 Prompt-robustness benchmark (Table 3)

A **300-sample stratified subsample** of the 500-sample human-validated set
(100 per stratum above, `judge/scripts/llm_prompt_perturbation.py`, seed 42),
each evaluated by every judge × every prompt variant (4×3 = 12 runs of 300).
File: `eval_cache/prompt_robustness_benchmark.json`.

### 4.5 Fine-grained error-type annotation (Table 4 / Appendix B)

The 150 `agreed_critical` samples from §4.3, each broken into a fine-grained
error category (callsign / runway / altitude / heading / frequency /
clearance / navigation / weather / other) per judge, then compared against
human category labels. Files: `eval_cache/error_type_annotation_{wide,long}.csv`.

---

## 5. Prompt variants

Four judge-prompt variants (`judge/prompts/`), sharing the same binary output
schema (`Equivalent` vs. `Critical_Errors`) and ICAO/ATC-phraseology
equivalence rules, used as a controlled ablation over prompt design:

| Variant | Design | Used for |
|---|---|---|
| `cot_v3` | **Canonical.** Full structure: ICAO normalization table, worked examples, explicit entity-priority ordering, reasoning bounded inside `<reasoning>...</reasoning>` tags (stripped before JSON parsing) | All main-paper results (Table 1, 2, 4) |
| `compressed` | Same rules, minimal verbosity, no worked examples | Robustness ablation (Table 3) |
| `no_examples` | Full structure minus the worked examples | Robustness ablation (Table 3) |
| `lenient` | No predefined entity categories at all — pure semantic judgment | Robustness ablation (Table 3) |

Model-specific chat templating (Llama/Qwen/DeepSeek/Mistral/Gemma special
tokens) is applied automatically by `judge/prompts/build_prompt()`.

---

## 6. Metrics

| Metric | Formula / definition | Direction |
|---|---|---|
| **WER** | Standard corpus-level word error rate (Levenshtein over words), `jiwer` | lower = better |
| **WeightedWER (WWER)** | Same edit distance, but substitution/deletion of a rule-tagged *entity* token (callsign, runway, altitude, heading, frequency, flight level, squawk — `weighted_wer.py`) costs weight **3**; all other tokens cost 1. A cheap, deterministic, LLM-free baseline for "does entity-aware weighting alone fix the problem?" | lower = better |
| **BERTDist** | `1 − BERTScore F1` (RoBERTa-large embeddings, no baseline rescaling) | lower = better |
| **SemDist** | `1 − cosine_similarity` of mean-pooled RoBERTa-large sentence embeddings | lower = better |
| **CxER** | `Critical_Errors / (Critical_Errors + Equivalent)` over judge (or majority-vote ensemble) classifications; samples the judge failed to classify (`Parse_Failure`) are excluded from both numerator and denominator | lower = better |

CxER classification prompt gives the judge ICAO/ATC-phraseology equivalence
rules (e.g. "tree"=3, "niner"=9, "FL350"="flight level three five zero" are
equivalent) so that surface-form differences that preserve operational
meaning are **not** counted as errors — only a changed callsign digit,
altitude, heading, runway, direction, frequency, clearance, or waypoint is
`Critical_Errors`.

---

## 7. Results — what each table is and how it's computed

### Table 1 — The Finetuning Paradox (`analysis/table1_finetuning_paradox.py`)

**Question:** among the samples each standard metric rates as *high quality*,
does the proportion that are actually operationally critical change after
fine-tuning?

**Method:** for each metric M ∈ {WER, WeightedWER, BERTDist, SemDist} and
each (ASR family, corpus), compute the 25th-percentile threshold τ of the
**pretrained** model's M-distribution (i.e. the "best quartile" region that
metric calls high-confidence). Then, within that *same* threshold region,
measure the percentage of `Critical_Errors` samples (per the `cot_v3`
majority-vote CxER label) both before and after fine-tuning, and report the
percentage-point change Δ = post − pre.

**Reading it:** a large positive Δ means fine-tuning makes the metric's
"high-confidence" region *more* concentrated with critical errors — i.e. the
metric becomes a *less* trustworthy proxy for safety exactly as it improves.

Current values:

| Metric | Model | ATCO2 | ATCC | ATCOSim |
|---|---|---|---|---|
| Δ_WER | Whisper | +23.9 | +34.5 | +52.6 |
| Δ_WER | Parakeet | +14.2 | +22.7 | +31.9 |
| Δ_WeightedWER | Whisper | +27.3 | +38.0 | +49.5 |
| Δ_WeightedWER | Parakeet | +15.2 | +20.4 | +33.9 |
| Δ_BERTDist | Whisper | +19.7 | +40.0 | +51.4 |
| Δ_BERTDist | Parakeet | +8.2 | +17.1 | +30.9 |
| Δ_SemDist | Whisper | +19.1 | +38.2 | +45.1 |
| Δ_SemDist | Parakeet | +13.3 | +23.1 | +35.6 |

The WeightedWER row exists specifically to rule out the cheap fix ("just
weight entity tokens more") — it fails almost identically to plain WER,
motivating CxER's meaning-aware judgment instead of a fixed lexical weight.

### Table 2 — CxER Validation (`analysis/table2_inter_rater_agreement.py`)

**Question:** does CxER's judge-ensemble label agree with human aviation
experts well enough to trust it as ground truth for Table 1?

**Method:** Cohen's κ between each individual judge and each human annotator
on the 500-sample validation set (§4.3), averaged over the two annotators;
the same for the majority-vote ensemble vs. each human; Human1-vs-Human2 κ as
the practical agreement ceiling; F1 of each judge/ensemble against the human
majority label ("Gold"). Because the validation set intentionally oversamples
disagreement cases (harder than the natural corpus distribution), a
population-reweighted κ is also reported, derived from inverse-probability
weights computed directly from the full annotated corpus's natural
(agreed-critical, agreed-equivalent, disagreement) proportions — currently
10,059 / 5,836 / 6,066 (computed dynamically by
`table2_inter_rater_agreement.py` from `annotations/`, so it stays correct
as the underlying data changes, e.g. after a JSON-parse-failure recovery
pass like the one applied in this release.

Current values, with 95% bootstrap CIs (1000 resamples, resampling the 500
validation rows with the two-annotator pairing preserved each replicate):

| | Cohen's κ (avg. over 2 annotators) | 95% CI | F1 vs. Gold |
|---|---|---|---|
| Human1 vs. Human2 (ceiling) | 0.650 | [0.583, 0.714] | — |
| Llama | 0.529 | [0.459, 0.591] | 0.825 |
| Qwen | 0.488 | [0.425, 0.551] | 0.766 |
| DeepSeek | 0.592 | [0.531, 0.648] | 0.845 |
| **Majority-vote ensemble** | **0.593** (population-reweighted: 0.657) | [0.537, 0.651] (reweighted: [0.603, 0.706]) | **0.866** |

None of the CIs cross zero — agreement is well above chance throughout,
including for the individual judges.

### Table 3 — Prompt Robustness (`analysis/table3_prompt_robustness.py`)

**Question:** is CxER a stable signal, or an artifact of one particular
prompt's wording?

**Method:** each of the 3 judges classifies the same 300-sample benchmark
(§4.4) under all 4 prompt variants (§5). F1 against the human label per
(judge, prompt) cell; Fleiss' κ across the 4 prompt variants per judge
(agreement of a judge with *itself* under different phrasings — the
robustness signal).

Current values (100% coverage):

| Judge | cot_v3 | compressed | no_examples | lenient | mean ± std | Fleiss' κ (cross-prompt) |
|---|---|---|---|---|---|---|
| Llama | 0.908 | 0.891 | 0.833 | 0.836 | 0.867 ± 0.033 | 0.786 |
| Qwen | 0.916 | 0.874 | 0.884 | 0.895 | 0.892 ± 0.016 | 0.849 |
| DeepSeek | 0.841 | 0.893 | 0.844 | 0.882 | 0.865 ± 0.023 | 0.694 |

### Table 4 / Appendix B — Fine-grained error-type agreement (`analysis/error_type_analysis.py`)

**Question:** for samples all judges and the human annotator agree are
*critical*, do they agree on **what kind** of error it is (a callsign
error vs. a runway error vs. ...)?

**Method:** on the 150 `agreed_critical` samples (§4.5), each judge's
finest-priority entity-type label (callsign > runway > altitude > frequency
> heading > direction > clearance > weather > navigation > other, in that
priority order when a sample has multiple errors) is compared against the
human label per class, one-vs-rest, via Cohen's κ / accuracy / precision /
recall / F1. Class frequencies (multi-label — a sample can trigger more than
one category, 73/150 do):

| Class | Share of critical samples |
|---|---|
| callsign | 70.5% |
| other | 32.9% |
| runway | 18.1% |
| clearance | 16.8% |
| navigation | 13.4% |
| weather | 8.1% |
| frequency | 6.7% |
| altitude | 1.3% |
| heading | 1.3% |

`altitude` and `heading` have too little support (2 samples each) for a
stable κ estimate — reported for completeness, not as a strong claim.

### Figures

- **Figure 1** (`figure1_wer_scatter.py`): per-sample WER vs. CxER
  classification scatter, pretrained vs. fine-tuned, with the τ threshold
  band — the visual counterpart of Table 1's mechanism.
- **Figure 2** (`figure2_metric_comparison.py`): grouped bar chart of the
  percentage improvement in each standard metric vs. CxER, pretrained →
  fine-tuned, across all (model, corpus) combinations.
  

## 8. Reproducing the results

All pre-computed judge outputs and annotation data are included — steps 0–1
need multi-GPU hardware and are optional; steps 2–5 run on a standard
workstation against the shipped data.

```bash
# Step 0 — fine-tune ASR models (optional; needs 1 GPU, see §3.2 for hyperparameters)
python asr/scripts/train_whisper.py --model openai/whisper-medium --dataset combined --hf_user your-hf-username
python asr/scripts/train_parakeet.py --model nvidia/parakeet-tdt-0.6b-v3 --dataset combined --hf_user your-hf-username
python asr/scripts/asr_inference.py --model asr/checkpoints/whisper-medium-combined --model_type whisper --gpu 0
python asr/scripts/asr_inference.py --model asr/checkpoints/parakeet-tdt-0.6b-v3-combined/parakeet-tdt-0.6b-v3-combined.nemo --model_type parakeet --gpu 0

# Step 1 — run the LLM judges (optional; needs 4x GPU per judge, ~22k samples/judge; see §3.3)
python judge/scripts/llm_judge.py

# Step 2 — compute WER / WeightedWER / BERTDist / SemDist / CxER (CPU is fine; GPU speeds up BERTScore/SemDist)
python evaluate_asr.py

# Step 3 — reproduce every table and figure in §7
python analysis/figure1_wer_scatter.py
python analysis/figure2_metric_comparison.py
python analysis/table1_finetuning_paradox.py
python analysis/table2_inter_rater_agreement.py
python analysis/table3_prompt_robustness.py
python analysis/error_type_analysis.py

# Step 4 — prompt-robustness experiment (optional; rebuilds/reruns the Table 3 benchmark from scratch)
python judge/scripts/llm_prompt_perturbation.py   # rebuild the 300-sample benchmark
python judge/scripts/prompt_perturbation.py        # run 4 prompt variants x 3 judges

# Step 5 — fine-grained error-type annotation (optional)
python judge/scripts/error_type_annotation.py
```

The `asr/scripts/*.py` scripts take CLI arguments (`--help` for the full
list). Everything under `judge/scripts/`, `evaluate_asr.py`, and
`analysis/*.py` has no CLI — paths and settings are constants at the top of
each file, pointing at the locations shown in §1; edit them directly to
change input/output paths.
