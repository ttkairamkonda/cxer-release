'''
Whisper Fine-Tuning Script for ATC Domain
==========================================
Usage:
    python asr/scripts/train_whisper.py
    python asr/scripts/train_whisper.py --model openai/whisper-small
    python asr/scripts/train_whisper.py --dataset atcosim
    python asr/scripts/train_whisper.py --model openai/whisper-large-v3 --gpu 1
'''

import os
import argparse
# argparse._ActionsContainer.add_argument
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["DATASETS_DISABLE_TORCHCODEC"] = "1"



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="openai/whisper-medium",
                        help="Any HuggingFace Whisper model ID")
    parser.add_argument("--dataset", default="combined",
                        choices=["combined", "atcosim", "uwb_atcc"])
    parser.add_argument("--gpu",     default="4")
    parser.add_argument("--output",  default="asr/checkpoints")
    parser.add_argument("--hf_user", default="your-hf-username",
                        help="HuggingFace username for pushing model")
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]     = args.gpu
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import random
import numpy as np
import torch
import wandb

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path

from datasets import (
    load_dataset, load_from_disk, concatenate_datasets,
    DatasetDict, Audio
)
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from whisper_normalizer.english import EnglishTextNormalizer
from huggingface_hub import login
import evaluate

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEED           = 42
LANGUAGE       = "en"
TASK           = "transcribe"
BATCH_SIZE     = 16
GRAD_ACCUM     = 8
LEARNING_RATE  = 1e-5
WARMUP_STEPS   = 1000
NUM_EPOCHS     = 20
EARLY_STOP_PAT = 3
MAX_LABEL_LEN  = 225

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
            {"path": "Jzuluaga/uwb_atcc",       "split": "test"}
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

# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

whisper_normalizer = EnglishTextNormalizer()

def normalize_text(text: str) -> str:
    if not text or not str(text).strip():
        return ""
    return whisper_normalizer(str(text))

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def normalize_schema(ds):
    if "sentence" in ds.column_names:
        ds = ds.rename_column("sentence", "text")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    keep = {"audio", "text"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def load_splits(cfg_list: list):
    splits = []
    for cfg in cfg_list:
        ds = load_dataset(cfg["path"])[cfg["split"]]
        ds = normalize_schema(ds)
        splits.append(ds)
    if len(splits) == 1:
        return splits[0]
    return concatenate_datasets(splits).shuffle(seed=SEED)


def get_dataset(dataset_name: str, disk_dir: str) -> DatasetDict:
    snapshot_path = Path(disk_dir) / f"whisper_{dataset_name}"
    if snapshot_path.exists():
        print(f"Loading from disk: {snapshot_path}")
        return load_from_disk(str(snapshot_path))

    print(f"Downloading: {dataset_name} ...")
    cfg      = DATASET_CONFIGS[dataset_name]
    train_ds = load_splits(cfg["train"])
    test_ds  = load_splits(cfg["test"])
    dd       = DatasetDict({"train": train_ds, "test": test_ds})
    dd.save_to_disk(str(snapshot_path))
    print(f"Saved to disk: {snapshot_path}")
    return dd

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def get_prepare_fn(processor):
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(
            normalize_text(batch["text"])
        ).input_ids
        return batch
    return prepare_dataset


def is_valid(example):
    return (
        example["audio"] is not None
        and example["audio"].get("array") is not None
        and len(example["audio"]["array"]) > 0
        and example["text"] is not None
        and str(example["text"]).strip() != ""
    )

# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLATOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

# ─────────────────────────────────────────────────────────────────────────────
# METRICS  — only used during training for live monitoring
# ─────────────────────────────────────────────────────────────────────────────

wer_metric = evaluate.load("wer")

def make_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm  = [normalize_text(s) for s in pred_str]
        label_norm = [normalize_text(s) for s in label_str]

        print("\nSample predictions:")
        for p, r in zip(pred_norm[:3], label_norm[:3]):
            print(f"  PRED: {p}")
            print(f"  REF : {r}\n")

        wer = 100 * wer_metric.compute(
            predictions=pred_norm, references=label_norm
        )
        return {"wer": wer}
    return compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    model_short  = args.model.split("/")[-1]
    dataset_name = args.dataset
    run_name     = f"{model_short}-{dataset_name}"
    output_dir   = f"{args.output}/{run_name}"
    disk_dir     = "data/processed"

    # ── Logins ────────────────────────────────────────────────────────────
#    login()       # HuggingFace — will prompt for token once
#    wandb.login() # WandB — for live epoch monitoring only

    os.environ["WANDB_PROJECT"] = "EMNLP2026_ASR"

    print(f"\n{'='*60}")
    print(f"  Model    : {args.model}")
    print(f"  Dataset  : {dataset_name}")
    print(f"  GPU      : {args.gpu}")
    print(f"  Output   : {output_dir}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    dataset    = get_dataset(dataset_name, disk_dir)
    processor  = WhisperProcessor.from_pretrained(
        args.model, language=LANGUAGE, task=TASK
    )
    dataset    = dataset.filter(is_valid, num_proc=1)
    prepare_fn = get_prepare_fn(processor)
    dataset    = dataset.map(
        prepare_fn,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )
    print(dataset)

    # ── Model ─────────────────────────────────────────────────────────────
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to("cuda")
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.suppress_tokens = []
    model.config.use_cache       = False

    # ── Training args ─────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir                    = output_dir,
        run_name                      = run_name,
        per_device_train_batch_size   = BATCH_SIZE,
        gradient_accumulation_steps   = GRAD_ACCUM,
        learning_rate                 = LEARNING_RATE,
        warmup_steps                  = WARMUP_STEPS,
        num_train_epochs              = NUM_EPOCHS,
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        fp16                          = True,
        eval_strategy                 = "epoch",
        save_strategy                 = "epoch",
        per_device_eval_batch_size    = BATCH_SIZE,
        predict_with_generate         = True,
        generation_max_length         = MAX_LABEL_LEN,
        logging_strategy              = "epoch",
        report_to                     = "wandb",
        label_names                   = ["labels"],
        remove_unused_columns         = False,
        load_best_model_at_end        = True,
        save_total_limit              = 2,
        metric_for_best_model         = "wer",
        greater_is_better             = False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        args            = training_args,
        model           = model,
        train_dataset   = dataset["train"],
        eval_dataset    = dataset["test"],
        data_collator   = DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics = make_compute_metrics(processor),
        tokenizer       = processor.feature_extractor,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PAT)],
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("Starting training...")
    start = time.time()
    trainer.train()
    print(f"Training complete in {(time.time()-start)/60:.1f} min")

    # ── Save locally ──────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    trainer.save_state()
    print(f"Model saved to {output_dir}")

    # ── Push to HuggingFace Hub ───────────────────────────────────────────
    hf_repo_id = f"{args.hf_user}/{run_name}"
    print(f"Pushing to HuggingFace Hub: {hf_repo_id} ...")
    model.push_to_hub(hf_repo_id,
                      commit_message=f"Fine-tuned {args.model} on {dataset_name}")
    processor.push_to_hub(hf_repo_id,
                          commit_message=f"Processor for {args.model} on {dataset_name}")
    print(f"Done! Check it out: https://huggingface.co/{hf_repo_id}")

    wandb.finish()


if __name__ == "__main__":
    main()