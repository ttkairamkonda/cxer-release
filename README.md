# CxER: Contextual Error Rate for ASR Evaluation

Code and data for **"Rethinking Evaluation in Automatic Speech Recognition"** (EMNLP 2026).

CxER is an LLM ensemble framework that evaluates whether ASR outputs preserve **operational meaning** rather than surface similarity. Applied to Air Traffic Control (ATC) transcription, it exposes a *Finetuning Paradox*: standard metrics (WER, BERTScore, SemDist) improve after fine-tuning while operationally critical errors concentrate precisely where those metrics signal confidence.

> **Note on naming:** In the codebase you will see `CCER` (Contextual Critical Error Rate). This was our internal name during development — `CER` was already taken by Character Error Rate, so we used `CCER` to avoid collision. The paper uses `CxER` throughout. The two refer to the same metric.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Three requirement files are provided due to hard version conflicts between NeMo (`torch==2.2`) and vLLM (`torch>=2.9`). Use the appropriate one per environment:

| File | Use for |
|---|---|
| `requirements.txt` | CxER evaluation pipeline + Weighted WER baseline |
| `asr_requirements.txt` | Whisper fine-tuning |
| `nvidia_requirements.txt` | Parakeet fine-tuning (NeMo) |

For gated HuggingFace models: `export HF_TOKEN=your_token_here`

---

## Reproducing the Results

All pre-computed judge outputs and annotation data are included. **Steps 0–1 are optional** (require multi-GPU hardware). Steps 2–3 run on a standard workstation.

### Step 0 — Fine-tune ASR models *(optional)*

```bash
python asr/scripts/train_whisper.py --model openai/whisper-medium --dataset combined --hf_user your-hf-username
python asr/scripts/train_parakeet.py --model nvidia/parakeet-tdt-0.6b-v3 --dataset combined --hf_user your-hf-username
python asr/scripts/asr_inference.py --model asr/checkpoints/whisper-medium-combined --model_type whisper --gpu 0
python asr/scripts/asr_inference.py --model asr/checkpoints/parakeet-tdt-0.6b-v2-combined.nemo --model_type parakeet --gpu 0
```

### Step 1 — Run LLM judges *(optional, requires 4× A100 per model)*

```bash
python judge/scripts/llm_judge.py
```

Judges: `Llama-3.3-70B-Instruct`, `Qwen2.5-72B-Instruct`, `DeepSeek-R1-Distill-Llama-70B`. Resume support is built in.

### Step 2 — Compute metrics

```bash
python evaluate_asr.py
```

Produces per-file WER, BERTDist, SemDist, and CxER. Results cached to `eval_cache/`; re-running is a no-op.

### Step 3 — Reproduce all tables and figures

```bash
python analysis/figure1_wer_scatter.py
python analysis/figure2_metric_comparison.py
python analysis/table1_finetuning_paradox.py
python analysis/table2_inter_rater_agreement.py
python analysis/table3_prompt_robustness.py
python analysis/table4_error_type_analysis.py
```

All scripts read from pre-populated `eval_cache/` and `annotations/`.

### Step 4 — Prompt robustness experiment *(optional)*

```bash
python judge/scripts/llm_prompt_perturbation.py   # build 300-sample benchmark
python judge/scripts/prompt_perturbation.py        # run 4 variants × 3 models
```

### Step 5 — Error type annotation *(optional)*

```bash
python judge/scripts/error_type_annotation.py
```

Human-annotated results are pre-included in `eval_cache/error_type_annotation_long_annotated.csv`.

---

## Repository Structure

```
cxer-release/
├── evaluate_asr.py                        # Step 2: compute all metrics (WER, WeightedWER, BERTDist, SemDist, CxER)
├── weighted_wer.py                        # Rule-based ATC entity tagger + Weighted WER (no LLM)
├── analysis/
│   ├── figure1_wer_scatter.py
│   ├── figure2_metric_comparison.py
│   ├── table1_finetuning_paradox.py
│   ├── table2_inter_rater_agreement.py
│   ├── table3_prompt_robustness.py
│   └── table4_error_type_analysis.py
├── asr/scripts/                           # train_whisper.py, train_parakeet.py, asr_inference.py
├── judge/
│   ├── prompts/                           # cot_v3.py (canonical), compressed.py, no_examples.py, lenient.py
│   └── scripts/                           # llm_judge.py, prompt_perturbation.py, error_type_annotation.py
├── data/raw/                              # 12 ASR transcript JSONs (2 models × 2 variants × 3 datasets)
├── annotations/{llama_70b,qwen_72b,deepseek_70b}/   # 36 judge output JSONs
├── eval_cache/                            # Cached metrics, SME annotations, robustness benchmark
├── prompt_robustness_outputs/             # Raw LLM outputs for robustness experiment
└── requirements.txt / asr_requirements.txt / nvidia_requirements.txt
```

---

## Data

**Raw transcripts** (`data/raw/`): 12 JSON files — `{model}_{variant}_{dataset}_predictions.json`. Models: `whisper-medium`, `parakeet-tdt-0.6b-v3`. Variants: `pretrained`, `combined`. Datasets: `atcosim_test`, `uwb_atcc_test`, `atco2_ood`.

**Judge outputs** (`annotations/`): same 12-file structure per judge, each record extended with `contextual_status`, `operational_safety`, `transcription_errors`, and `WER`.

**Human annotations** (`eval_cache/annotation_sheet_shared.xlsx`): 500 samples annotated by two aviation subject-matter experts. Population counts for reweighting: 9,988 agreed-critical, 4,729 agreed-equivalent, 5,822 disagreed.

**Prompt variants** (`judge/prompts/`): `cot_v3` (canonical, used for all main results), `compressed`, `no_examples`, `lenient`.

---

