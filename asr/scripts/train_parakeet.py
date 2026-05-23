'''
Parakeet (NeMo) Fine-Tuning Script for ATC Domain
==================================================
Usage:
    python asr/scripts/train_parakeet.py
    python asr/scripts/train_parakeet.py --model nvidia/parakeet-tdt-0.6b-v2
    python asr/scripts/train_parakeet.py --dataset atcosim --gpu 1
'''

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="nvidia/parakeet-tdt-0.6b-v3",
                        help="Any NeMo-compatible Parakeet model ID")
    parser.add_argument("--dataset", default="combined",
                        choices=["combined", "atcosim", "uwb_atcc"])
    parser.add_argument("--gpu",     default="6")
    parser.add_argument("--output",  default="asr/checkpoints")
    parser.add_argument("--hf_user", default="your-hf-username",
                        help="HuggingFace username for pushing model")
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]     = args.gpu
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
logging.getLogger("nemo").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import json
import random
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
import wandb

from datasets import load_dataset, load_from_disk, Audio, concatenate_datasets
from huggingface_hub import login, HfApi
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor, ModelCheckpoint, EarlyStopping
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEED           = 42
TRAIN_BATCH    = 16
VAL_BATCH      = 8
NUM_EPOCHS     = 20
GRAD_ACCUM     = 4
GRAD_CLIP      = 1.0
LEARNING_RATE  = 1e-5
WARMUP_STEPS   = 1000
WEIGHT_DECAY   = 0.01
BETAS          = [0.9, 0.98]
EARLY_STOP_PAT = 5
MANIFEST_DIR   = "data/manifests"
MIN_DURATION   = 0.1
MAX_DURATION   = 30.0

DATASET_CONFIGS = {
    "atcosim": {
        "train": [{"path": "Jzuluaga/atcosim_corpus", "split": "train"}],
        "test":  [{"path": "Jzuluaga/atcosim_corpus", "split": "test"}],
    },
    "uwb_atcc": {
        "train": [{"path": "Jzuluaga/uwb_atcc", "split": "train"}],
        "test":  [{"path": "Jzuluaga/uwb_atcc", "split": "test"}],
    },
    "combined": {
        "train": [
            {"path": "Jzuluaga/atcosim_corpus", "split": "train"},
            {"path": "Jzuluaga/uwb_atcc",       "split": "train"},
        ],
        "test": [
            {"path": "Jzuluaga/atcosim_corpus", "split": "test"},
            {"path": "Jzuluaga/uwb_atcc",       "split": "test"},
        ],
    },
}

OOD_CONFIG = {"path": "Jzuluaga/atco2_corpus_1h", "split": "test"}

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
os.makedirs(MANIFEST_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZATION  — uppercase for Parakeet
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text) -> str:
    if not text or not str(text).strip():
        return ""
    return str(text).strip().upper()

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  — HF dataset → NeMo manifest records
# ─────────────────────────────────────────────────────────────────────────────

def hf_split_to_records(hf_path: str, split: str, dataset_name: str) -> list:
    """
    Download one HF split, write audio arrays to temp wavs,
    return list of NeMo manifest dicts.
    Snapshots the record list to disk as JSON to avoid re-downloading.
    """
    snapshot_path = Path(MANIFEST_DIR) / f"{dataset_name}_{split}_records.json"
    if snapshot_path.exists():
        print(f"  Loading records from disk: {snapshot_path}")
        with open(snapshot_path) as f:
            return json.load(f)

    print(f"  Downloading {hf_path} [{split}] ...")
    ds      = load_dataset(hf_path)[split]
    ds      = ds.cast_column("audio", Audio(sampling_rate=16000))
    tmp_dir = Path(MANIFEST_DIR) / f"audio_{dataset_name}_{split}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for i, item in enumerate(ds):
        text = normalize_text(
            item.get("text") or item.get("sentence", "")
        )
        if not text:
            continue

        audio = item.get("audio")
        if audio is None:
            continue

        # Write in-memory array to wav
        array = np.array(audio["array"], dtype=np.float32)
        sr    = audio.get("sampling_rate", 16000)
        if len(array) == 0:
            continue

        wav_path = os.path.join(tmp_dir, f"sample_{i}.wav")
        sf.write(wav_path, array, sr)

        try:
            y, _     = librosa.load(wav_path, sr=16000)
            duration = float(len(y)) / 16000
            if duration < MIN_DURATION or duration > MAX_DURATION:
                continue
            records.append({
                "audio_filepath": wav_path,
                "duration":       duration,
                "text":           text,
            })
        except Exception as e:
            print(f"  Warning: skipping sample {i} — {e}")

    # Snapshot records list so we don't re-download next run
    with open(snapshot_path, "w") as f:
        json.dump(records, f)
    print(f"  Saved {len(records)} records → {snapshot_path}")
    return records


def load_records(cfg_list: list, split_label: str) -> list:
    """Load and concatenate records from multiple dataset configs."""
    all_records = []
    for cfg in cfg_list:
        dataset_tag = cfg["path"].split("/")[-1]
        records = hf_split_to_records(cfg["path"], cfg["split"], dataset_tag)
        all_records.extend(records)
    if split_label == "train":
        random.shuffle(all_records)
    return all_records

# ─────────────────────────────────────────────────────────────────────────────
# MANIFEST WRITING
# ─────────────────────────────────────────────────────────────────────────────

def write_manifest(records: list, path: str) -> int:
    valid = 0
    with open(path, "w") as f:
        for item in records:
            if not os.path.exists(item["audio_filepath"]):
                continue
            text = normalize_text(item.get("text", ""))
            if not text:
                continue
            entry = {
                "audio_filepath": item["audio_filepath"],
                "duration":       item.get("duration", 0.0),
                "text":           text,
            }
            f.write(json.dumps(entry) + "\n")
            valid += 1
    print(f"  Wrote {valid} entries → {path}")
    return valid

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    model_short  = args.model.split("/")[-1]
    dataset_name = args.dataset
    run_name     = f"{model_short}-{dataset_name}"
    output_dir   = Path(args.output) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    login()
    wandb.login()
    os.environ["WANDB_PROJECT"] = "EMNLP2026_ASR"

    print(f"\n{'='*60}")
    print(f"  Model    : {args.model}")
    print(f"  Dataset  : {dataset_name}")
    print(f"  GPU      : {args.gpu}")
    print(f"  Output   : {output_dir}")
    print(f"{'='*60}\n")

    # ── Load records & write manifests ────────────────────────────────────
    cfg = DATASET_CONFIGS[dataset_name]

    print("Loading train records...")
    train_records = load_records(cfg["train"], "train")
    print("Loading test records...")
    test_records  = load_records(cfg["test"],  "test")

    train_manifest = os.path.join(MANIFEST_DIR, f"train_{dataset_name}.json")
    test_manifest  = os.path.join(MANIFEST_DIR, f"test_{dataset_name}.json")

    n_train = write_manifest(train_records, train_manifest)
    n_test  = write_manifest(test_records,  test_manifest)
    print(f"Train: {n_train}  |  Test: {n_test}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading {args.model} ...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    asr_model.gradient_checkpointing = True
    asr_model.log_prediction         = False

    # ── Data loader configs ───────────────────────────────────────────────
    train_ds_cfg = OmegaConf.create({
        "manifest_filepath": train_manifest,
        "sample_rate":       16000,
        "batch_size":        TRAIN_BATCH,
        "shuffle":           True,
        "num_workers":       4,
        "pin_memory":        True,
        "text_field":        "text",
        "min_duration":      MIN_DURATION,
        "max_duration":      MAX_DURATION,
        "trim_silence":      False,
        "load_audio":        True,
        "shuffle_seed":      SEED,
        "shard_strategy":    "scatter",
    })
    test_ds_cfg = OmegaConf.create({
        "manifest_filepath": test_manifest,
        "sample_rate":       16000,
        "batch_size":        VAL_BATCH,
        "shuffle":           False,
        "num_workers":       2,
        "pin_memory":        True,
        "text_field":        "text",
        "min_duration":      MIN_DURATION,
        "max_duration":      MAX_DURATION,
        "trim_silence":      False,
        "load_audio":        True,
    })

    asr_model.setup_training_data(train_data_config=train_ds_cfg)
    asr_model.setup_validation_data(val_data_config=test_ds_cfg)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # asr_model.setup_optimization(OmegaConf.create({
    #     "name":         "adamw",
    #     "lr":           LEARNING_RATE,
    #     "betas":        BETAS,
    #     "weight_decay": WEIGHT_DECAY,
    # }))
    asr_model.setup_optimization(OmegaConf.create({
    "name":          "adamw",
    "lr":            LEARNING_RATE,
    "betas":         BETAS,
    "weight_decay":  WEIGHT_DECAY,
    "sched": {
        "name":        "WarmupAnnealing",
        "warmup_steps": WARMUP_STEPS,
    }
    }))
    # ── Callbacks & trainer ───────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath    = str(output_dir),
        save_top_k = 2,
        monitor    = "val_wer",
        mode       = "min",
        save_last  = True,
        filename   = "{epoch}-{val_wer:.4f}",
    )
    wandb_logger = WandbLogger(
        project  = "EMNLP2026_ASR",
        name     = run_name,
        save_dir = "./wandb_logs",
    )

    trainer = pl.Trainer(
        devices                 = 1,
        accelerator             = "gpu",
        precision               = "bf16-mixed",
        max_epochs              = NUM_EPOCHS,
        gradient_clip_val       = GRAD_CLIP,
        accumulate_grad_batches = GRAD_ACCUM,
        logger                  = wandb_logger,
        log_every_n_steps       = 10,
        enable_progress_bar     = True,
        check_val_every_n_epoch = 1,
        callbacks               = [
            LearningRateMonitor(logging_interval="epoch"),
            checkpoint_cb,
            EarlyStopping(
                monitor  = "val_wer",
                patience = EARLY_STOP_PAT,
                mode     = "min",
                verbose  = True,
            ),
            pl.callbacks.TQDMProgressBar(refresh_rate=10),
        ],
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("Starting training...")
    start = time.time()
    asr_model.set_trainer(trainer)
    trainer.fit(asr_model)
    print(f"Training complete in {(time.time()-start)/60:.1f} min")

    # ── Save .nemo ────────────────────────────────────────────────────────
    nemo_filename = str(output_dir / f"{run_name}.nemo")
    asr_model.save_to(nemo_filename)
    print(f"Model saved → {nemo_filename}")

    # ── Push to HuggingFace Hub ───────────────────────────────────────────
    hf_repo_id = f"{args.hf_user}/{run_name}"
    print(f"Pushing to HuggingFace Hub: {hf_repo_id} ...")
    api = HfApi()
    api.create_repo(repo_id=hf_repo_id, exist_ok=True)
    api.upload_file(
        path_or_fileobj = nemo_filename,
        path_in_repo    = Path(nemo_filename).name,
        repo_id         = hf_repo_id,
        commit_message  = f"Fine-tuned {args.model} on {dataset_name}",
    )
    print(f"Done  https://huggingface.co/{hf_repo_id}")

    wandb.finish()


if __name__ == "__main__":
    main()