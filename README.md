# CxER: Contextual Error Rate for ASR Evaluation

Code and data for the paper **"Rethinking Evaluation in Automatic Speech Recognition"** (EMNLP 2026).

CxER is an LLM ensemble framework that evaluates whether ASR outputs preserve **operational meaning** rather than surface similarity. Applied to Air Traffic Control (ATC) transcription, it exposes a *Finetuning Paradox*: standard metrics (WER, BERTScore, SemDist) improve after fine-tuning while operationally critical errors concentrate precisely where those metrics signal confidence.

---

## Repository Structure

```
cxer-release/
│
├── evaluate_asr.py                  # Compute WER, BERTDist, SemDist, CxER from judge outputs
│
├── analysis/
│   ├── figure1_wer_scatter.py       # Figure 1 — per-utterance WER vs CxER label scatter
│   ├── figure2_metric_comparison.py # Figure 2 — grouped bar chart across all metrics
│   ├── table1_finetuning_paradox.py # Table 1 — 25th-percentile threshold analysis
│   ├── table2_inter_rater_agreement.py  # Table 2 / Appendix C — Cohen's κ, Fleiss' κ
│   ├── table3_prompt_robustness.py  # Table 3 — per-prompt F1 and Fleiss' κ
│   └── table4_error_type_analysis.py    # Table 4 / Appendix B — per-class κ, Krippendorff's α
│
├── asr/
│   └── scripts/
│       ├── train_whisper.py         # Fine-tune Whisper via HuggingFace Seq2SeqTrainer
│       ├── train_parakeet.py        # Fine-tune Parakeet via NeMo + PyTorch Lightning
│       └── asr_inference.py         # Run inference; produces data/raw/ JSON files
│
├── requirements.txt                 # Python deps for CxER evaluation pipeline
├── asr_requirements.txt             # Python deps for Whisper fine-tuning (HuggingFace stack)
├── nvidia_requirements.txt          # Python deps for Parakeet fine-tuning (NeMo stack)
│
├── data/
│   └── raw/                         # ASR transcripts — 12 files (2 models × 2 variants × 3 datasets)
│
├── judge/
│   ├── prompts/                     # Four prompt variants + chat-template formatter
│   │   ├── cot_v3.py                # Canonical CoT prompt (used for all main results)
│   │   ├── compressed.py
│   │   ├── no_examples.py
│   │   ├── lenient.py
│   │   └── utils.py                 # build_prompt() dispatcher with per-model chat templates
│   └── scripts/
│       ├── llm_judge.py             # Run CxER pipeline on all 12 raw ASR files
│       ├── prompt_perturbation.py   # Run 4 prompt variants × 3 models on 300-sample benchmark
│       ├── llm_prompt_perturbation.py  # Build the 300-sample stratified benchmark
│       └── error_type_annotation.py    # Prepare error-type annotation sheets for SMEs
│
├── annotations/                     # LLM judge outputs — 12 JSONs per judge (36 total)
│   ├── llama_70b/
│   ├── qwen_72b/
│   └── deepseek_70b/
│
├── eval_cache/                      # Cached metric results and annotation data
│   ├── annotation_sheet_shared.csv  # 500 SME-annotated samples (binary CxER labels)
│   ├── annotation_sheet_shared.xlsx # Same sheet with consensus + per-judge columns
│   ├── prompt_robustness_benchmark.json  # 300-sample stratified benchmark
│   ├── {llama,qwen,deepseek,all}_results.csv  # Cached WER/BERTDist/SemDist/CxER per file
│   ├── error_type_annotation_wide.csv     # Per-class binary columns for κ computation
│   ├── error_type_annotation_long.csv     # Long format for SME annotation
│   └── error_type_annotation_long_annotated.csv  # Annotated version
│
├── prompt_robustness_outputs/       # Raw LLM outputs for robustness experiment
│   ├── cot_v3/   {llama,qwen,deepseek}.json
│   ├── compressed/
│   ├── no_examples/
│   └── lenient/
│
└── prompt_robustness_analysis/      # Derived CSVs and LaTeX tables (Table 3)
    ├── table1_f1_scores.{csv,tex}
    ├── table2_agreement.{csv,tex}
    └── ...
```

---

## Setup

### CxER evaluation environment

**Python 3.10+** required. Install dependencies:

```bash
pip install vllm transformers torch bert-score jiwer \
            pandas numpy matplotlib seaborn scikit-learn \
            statsmodels scipy krippendorff openpyxl \
            python-dotenv huggingface_hub tqdm
```

Set your Hugging Face token (needed to load gated models):

```bash
export HF_TOKEN=your_token_here
```

LLM judges used: `meta-llama/Llama-3.3-70B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`. Each requires ~4× A100 80GB GPUs with tensor parallelism.

### ASR fine-tuning environments

Three separate requirement files are provided because the three environments have hard version conflicts (NeMo requires `torch==2.2` and `numpy<2.0`; vLLM requires `torch>=2.9`):

| File | Environment | Key constraint |
|---|---|---|
| `requirements.txt` | CxER judges + evaluation | `torch==2.9`, `numpy==2.2.6` |
| `asr_requirements.txt` | Whisper fine-tuning | `torch==2.9`, HuggingFace stack |
| `nvidia_requirements.txt` | Parakeet fine-tuning | `torch==2.2`, `numpy==1.26.4`, NeMo |

```bash
# CxER evaluation (steps 1–5 of the reproduction guide)
pip install -r requirements.txt

# Whisper fine-tuning only (step 0, separate venv)
pip install -r asr_requirements.txt

# Parakeet fine-tuning only (step 0, separate venv)
pip install -r nvidia_requirements.txt
```

---

## Reproducing the Results

All pre-computed judge outputs and annotation data are included. You can skip directly to analysis (step 3) without re-running the LLMs. Steps 0–1 require multi-GPU cluster hardware; steps 2–5 run on a standard workstation.

### Step 0 — Fine-tune ASR models and generate transcripts (optional)

The raw ASR transcripts in `data/raw/` are pre-included. Run this step only if you want to reproduce the fine-tuning from scratch.

**Whisper Medium** (HuggingFace `Seq2SeqTrainer`, requires 1× A100):
```bash
python asr/scripts/train_whisper.py --model openai/whisper-medium --dataset combined \
    --hf_user your-hf-username
```

**Parakeet TDT 0.6B** (NeMo + PyTorch Lightning, requires 1× A100):
```bash
python asr/scripts/train_parakeet.py --model nvidia/parakeet-tdt-0.6b-v3 --dataset combined \
    --hf_user your-hf-username
```

After fine-tuning, run inference to produce the 12 raw transcript JSONs:
```bash
python asr/scripts/asr_inference.py --model asr/checkpoints/whisper-medium-combined \
    --model_type whisper --gpu 0
python asr/scripts/asr_inference.py \
    --model asr/checkpoints/parakeet-tdt-0.6b-v2-combined/parakeet-tdt-0.6b-v2-combined.nemo \
    --model_type parakeet --gpu 0
```

Outputs are written to `data/raw/` in the format `{model}_{variant}_{dataset}_predictions.json`.

### Step 1 — Run CxER judges on raw ASR transcripts

Reads from `data/raw/`, writes to `annotations/{llama_70b,qwen_72b,deepseek_70b}/`.

```bash
python judge/scripts/llm_judge.py
```

Each of the 3 models is loaded, run over all 12 input files, and unloaded before the next model is loaded. Resume support is built in — already-completed output files are skipped.

### Step 2 — Compute baseline metrics

Reads from `annotations/`, caches to `eval_cache/`.

```bash
python evaluate_asr.py
```

Produces per-file WER, BERTDist (1 − BERTScore F1), SemDist, and CxER for all model/variant/dataset combinations. Results are cached to CSV; re-running is a no-op if the cache exists.

### Step 3 — Analysis, tables, and figures

Each result in the paper has a dedicated script under `analysis/`. Run from the repo root:

```bash
python analysis/figure1_wer_scatter.py        # Figure 1  → eval_plots/wer_whisper_only_atcosim_test_vertical.png
python analysis/figure2_metric_comparison.py  # Figure 2  → eval_plots/ccer_improvement_gap.png
python analysis/table1_finetuning_paradox.py  # Table 1   — prints Finetuning Paradox threshold analysis
python analysis/table2_inter_rater_agreement.py  # Table 2 / Appendix C — Cohen's κ, Fleiss' κ
python analysis/table3_prompt_robustness.py   # Table 3   → prompt_robustness_analysis/
python analysis/table4_error_type_analysis.py # Table 4 / Appendix B — per-class Krippendorff's α
```

All scripts read from `eval_cache/` and `annotations/` which are pre-populated. Re-running is safe.

### Step 4 — Prompt robustness experiment

First build the 300-sample benchmark (or use the pre-built one in `eval_cache/`):

```bash
python judge/scripts/llm_prompt_perturbation.py
```

Then run each model across all 4 prompt variants:

```bash
python judge/scripts/prompt_perturbation.py
```

Outputs written to `prompt_robustness_outputs/{cot_v3,compressed,no_examples,lenient}/`. Analysis is in `analysis/table3_prompt_robustness.py`.

### Step 5 — Error type annotation (fine-grained)

Prepare annotation sheets for human review of error categories:

```bash
python judge/scripts/error_type_annotation.py
```

Writes `eval_cache/error_type_annotation_{wide,long}.csv`. Human-annotated results are pre-included in `error_type_annotation_long_annotated.csv`.

---

## Data

### Raw ASR Transcripts (`data/raw/`)

Twelve JSON files, one per model/variant/dataset combination. Filename format:

```
{model}_{variant}_{dataset}_predictions.json
```

- **Models:** `whisper-medium`, `parakeet-tdt-0.6b-v3`
- **Variants:** `pretrained`, `combined` (fine-tuned on ATCOSIM + UWB-ATCC)
- **Datasets:** `atcosim_test`, `uwb_atcc_test`, `atco2_ood`

Each file is a JSON list of `{"reference": "...", "hypothesis": "..."}` pairs.

### Judge Outputs (`annotations/`)

Same 12-file structure per judge. Each record extends the raw pair with:

```json
{
  "reference": "...",
  "hypothesis": "...",
  "contextual_status": "Equivalent | Critical_Errors | Error",
  "meaning_preserved": true,
  "operational_safety": "safe | potentially_unsafe | unsafe",
  "explanation": "...",
  "transcription_errors": [
    {
      "entity_type": "callsign | runway | altitude | ...",
      "reference_value": "...",
      "transcribed_value": "...",
      "safety_risk": "high | medium | low",
      "explanation": "..."
    }
  ],
  "WER": 0.0
}
```

### Human Annotations (`eval_cache/annotation_sheet_shared.xlsx`)

500 samples annotated by two aviation subject-matter experts. Columns include binary CxER labels per judge and per human annotator, consensus labels, majority-vote LLM consensus, and `sample_type` (`agreed_critical`, `agreed_equivalent`, `disagreed`). Population counts for reweighting: 9,988 agreed-critical, 4,729 agreed-equivalent, 5,822 disagreed.

---

## Prompt Variants

Four prompt variants are included under `judge/prompts/`. All share the same JSON output schema.

| Variant | Description |
|---|---|
| `cot_v3` | **Canonical.** Full ICAO phonetics table, typed examples for all error classes, structured chain-of-thought reasoning inside `<reasoning>` tags. Used for all main results. |
| `compressed` | Same structure, examples and ICAO table condensed to minimal bullets. |
| `no_examples` | Full rules but no few-shot examples. |
| `lenient` | No structured taxonomy or examples; pure semantic judgment. Binary output only. |

---

## Citation

```bibtex
@inproceedings{cxer2026,
  title     = {Rethinking Evaluation in Automatic Speech Recognition},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026},
}
```

