'''
ASR Evaluation Script
=====================
Runs WER and CER on in-domain and OOD test sets for any fine-tuned model.
Saves results to eval/results/tables/asr_results.csv

Usage:
    # Whisper
    python asr/scripts/asr_inference.py \
        --model asr/checkpoints/whisper-medium-combined \
        --model_type whisper \
        --gpu 0

    # Wav2Vec2
    python asr/scripts/asr_inference.py \
        --model asr/checkpoints/wav2vec2-large-960h-combined \
        --model_type wav2vec2 \
        --gpu 0

    # Parakeet
    python asr/scripts/asr_inference.py \
        --model asr/checkpoints/parakeet-tdt-0.6b-v2-combined/parakeet-tdt-0.6b-v2-combined.nemo \
        --model_type parakeet \
        --gpu 0

    # Or directly from HuggingFace (replace {hf_user} with your username)
    python asr/scripts/asr_inference.py \
        --model {hf_user}/whisper-medium-combined \
        --model_type whisper \
        --gpu 0 \
        --batch_size 64 \
        --baseline \

        python asr/scripts/asr_inference.py \
    --model {hf_user}/parakeet-tdt-0.6b-v3-combined \
    --model_type parakeet \
    --gpu 0 \
    --batch_size 64 \
    --baseline \
'''

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,
                        help="Local checkpoint path or HuggingFace model ID")
    parser.add_argument("--model_type", required=True,
                        choices=["whisper", "wav2vec2", "parakeet"])
    parser.add_argument("--gpu",        default="0")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--output_dir", default="eval/results/tables")
    parser.add_argument("--baseline",   action="store_true",
                        help="Also evaluate the vanilla pretrained model for comparison")
    parser.add_argument("--baseline_model", default=None,
                        help="Pretrained model ID to use as baseline. "
                             "Defaults: whisper=openai/whisper-medium, "
                             "wav2vec2=facebook/wav2vec2-large-960h, "
                             "parakeet=nvidia/parakeet-tdt-0.6b-v2")
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import re
import json
import csv
from pathlib import Path
from datetime import datetime
import shutil
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Audio
from jiwer import wer, cer

# ─────────────────────────────────────────────────────────────────────────────
# TEST SET CONFIGS
# in-domain: ATCOSIM test + UWB-ATCC test
# OOD:       ATCO2 test
# ─────────────────────────────────────────────────────────────────────────────

TEST_SETS = {
    "atcosim_test": {
        "hf_path": "Jzuluaga/atcosim_corpus",
        "split":   "test",
        "domain":  "in-domain",
    },
    "uwb_atcc_test": {
        "hf_path": "Jzuluaga/uwb_atcc",
        "split":   "test",
        "domain":  "in-domain",
    },
    "atco2_ood": {
        "hf_path": "Jzuluaga/atco2_corpus_1h",
        "split":   "test",
        "domain":  "OOD",
    },
}

DISK_CACHE = "data/processed/eval"

# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZATION  — applied to both ref and hyp before WER
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_test_set(name: str, cfg: dict):
    """Load test set from disk cache or HuggingFace."""
    cache_path = Path(DISK_CACHE) / name
    if cache_path.exists():
        print(f"  Loading {name} from disk cache...")
        return load_from_disk(str(cache_path))

    print(f"  Downloading {name} from HuggingFace...")
    ds = load_dataset(cfg["hf_path"])[cfg["split"]]
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Normalize schema
    if "sentence" in ds.column_names:
        ds = ds.rename_column("sentence", "text")
    keep = {"audio", "text"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_path))
    return ds

# ─────────────────────────────────────────────────────────────────────────────
# MODEL INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_whisper(model_path: str, dataset, batch_size: int):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from whisper.normalizers import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer()

    print(f"  Loading Whisper model: {model_path}")
    BASE_WHISPER = "openai/whisper-medium"

    processor = WhisperProcessor.from_pretrained(BASE_WHISPER)
    # processor = WhisperProcessor.from_pretrained(model_path)
    model     = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval().cuda()

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en",
        task="transcribe"
    )

    references, hypotheses = [], []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Whisper inference"):
        batch   = dataset.select(range(i, min(i + batch_size, len(dataset))))
        arrays  = [x["array"] for x in batch["audio"]]
        texts   = batch["text"]

        inputs = processor(
            arrays,
            sampling_rate  = 16000,
            return_tensors = "pt",
            padding        = True,
        ).input_features.cuda()
        # inputs = processor(
        #     arrays,
        #     sampling_rate=16000,
        #     return_tensors="pt",
        # ).input_features.cuda()
        
        with torch.no_grad():
                predicted_ids = model.generate(
                    inputs,
                    forced_decoder_ids=forced_decoder_ids,
                )

        preds = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        for ref, hyp in zip(texts, preds):
            references.append(normalizer(str(ref)))
            hypotheses.append(normalizer(str(hyp)))

    return references, hypotheses


def run_wav2vec2(model_path: str, dataset, batch_size: int):
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

    print(f"  Loading Wav2Vec2 model: {model_path}")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model     = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval().cuda()

    references, hypotheses = [], []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Wav2Vec2 inference"):
        batch  = dataset.select(range(i, min(i + batch_size, len(dataset))))
        arrays = [x["array"] for x in batch["audio"]]
        texts  = batch["text"]

        inputs = processor(
            arrays,
            sampling_rate  = 16000,
            return_tensors = "pt",
            padding        = True,
        ).input_values.cuda()

        with torch.no_grad():
            logits    = model(inputs).logits
            pred_ids  = torch.argmax(logits, dim=-1)

        preds = processor.batch_decode(pred_ids)

        for ref, hyp in zip(texts, preds):
            references.append(normalize(str(ref)))
            hypotheses.append(normalize(str(hyp)))

    return references, hypotheses


def run_parakeet(model_path: str, dataset, batch_size: int):
    import nemo.collections.asr as nemo_asr
    import soundfile as sf
    import tempfile

    print(f"  Loading Parakeet model: {model_path}")

    if model_path.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(model_path)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_path)
    model.eval().cuda()

    # Parakeet needs audio file paths — write to temp wavs
    tmp_dir    = tempfile.mkdtemp(prefix="parakeet_eval_")
    audio_paths, references = [], []

    for i, item in enumerate(dataset):
        wav_path = os.path.join(tmp_dir, f"sample_{i}.wav")
        sf.write(wav_path, item["audio"]["array"], 16000)
        audio_paths.append(wav_path)
        references.append(normalize(str(item["text"])))

    # hypotheses_raw = model.transcribe(audio_paths, batch_size=batch_size)
    hypotheses_raw = model.transcribe(
        audio_paths,
        batch_size=batch_size,
        verbose=False
    )
    # transcribe() returns list or tuple depending on NeMo version
    if isinstance(hypotheses_raw[0], list):
        hypotheses_raw = hypotheses_raw[0]

    hypotheses = [normalize(str(h)) for h in hypotheses_raw]

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return references, hypotheses


def run_inference(model_path, model_type, dataset, batch_size):
    if model_type == "whisper":
        return run_whisper(model_path, dataset, batch_size)
    elif model_type == "wav2vec2":
        return run_wav2vec2(model_path, dataset, batch_size)
    elif model_type == "parakeet":
        return run_parakeet(model_path, dataset, batch_size)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    model_name = Path(args.model).name if os.path.exists(args.model) else args.model.split("/")[-1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "asr_results.csv"

    # Build list of models to evaluate
    # Each entry: (model_path, label, is_baseline)
    BASELINE_DEFAULTS = {
        "whisper":   "openai/whisper-medium",
        "wav2vec2":  "facebook/wav2vec2-large-960h",
        "parakeet":  "nvidia/parakeet-tdt-0.6b-v3",
    }

    models_to_eval = [
        (args.model, model_name, False),
    ]
    if args.baseline:
        baseline_model = args.baseline_model or BASELINE_DEFAULTS[args.model_type]
        baseline_name  = baseline_model.split("/")[-1] + "_baseline"
        models_to_eval.insert(0, (baseline_model, baseline_name, True))

    print(f"\n{'='*60}")
    print(f"  Model type : {args.model_type}")
    print(f"  GPU        : {args.gpu}")
    print(f"  Models     : {[m[1] for m in models_to_eval]}")
    print(f"{'='*60}\n")

    all_results = []

    for model_path, model_label, is_baseline in models_to_eval:
        print(f"\n{'='*60}")
        print(f"  Evaluating : {model_label}")
        print(f"  Baseline   : {is_baseline}")
        print(f"{'='*60}")

        for test_name, cfg in TEST_SETS.items():
            print(f"\n  Test set: {test_name} ({cfg['domain']})")

            dataset = load_test_set(test_name, cfg)
            dataset = dataset.filter(
                lambda x: x["audio"] is not None
                          and x["audio"].get("array") is not None
                          and len(x["audio"]["array"]) > 0
                          and bool(x.get("text", "").strip())
            )
            print(f"  Examples: {len(dataset)}")

            # dataset = dataset.shuffle(seed=42)
            # dataset = dataset.select(range(50))
            # if args.max_samples:
            #     dataset = dataset.select(range(min(args.max_samples, len(dataset))))


            references, hypotheses = run_inference(
                model_path, args.model_type, dataset, args.batch_size
            )

            wer_score = round(wer(references, hypotheses) * 100, 2)
            cer_score = round(cer(references, hypotheses) * 100, 2)
            print(f"  WER: {wer_score:.2f}%  |  CER: {cer_score:.2f}%")

            # Save predictions — only for fine-tuned model, not baseline
            # if not is_baseline:
            if True:
                preds_path = output_dir.parent / "raw" / f"{model_label}_{test_name}_predictions.json"
                preds_path.parent.mkdir(parents=True, exist_ok=True)
                with open(preds_path, "w") as f:
                    json.dump([
                        {"reference": r, "hypothesis": h}
                        for r, h in zip(references, hypotheses)
                    ], f, indent=2)
                print(f"  Predictions saved → {preds_path}")

            all_results.append({
                "model":      model_label,
                "model_type": args.model_type,
                "is_baseline": is_baseline,
                "test_set":   test_name,
                "domain":     cfg["domain"],
                "n_examples": len(dataset),
                "wer":        wer_score,
                "cer":        cer_score,
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

    # ── Write/append to CSV ───────────────────────────────────────────────
    file_exists = results_path.exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults appended → {results_path}")

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Model':<30} {'Test Set':<20} {'WER':>8} {'CER':>8}")
    print(f"{'─'*70}")
    for r in all_results:
        print(f"{r['model']:<30} {r['test_set']:<20} {r['wer']:>7.2f}% {r['cer']:>7.2f}%")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    main()