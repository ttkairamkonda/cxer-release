"""
ASR Evaluation Pipeline
Computes: Corpus WER, BERTScore F1, SemDist, Contextual Error Rate
Caches results to CSV. Plots separate figures per metric per judge.
"""

import os
import json
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import jiwer
import torch
from bert_score import score as bert_score_fn
from transformers import RobertaTokenizer, RobertaModel

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# ─── Config ───────────────────────────────────────────────────────────────────

_REPO = os.path.dirname(os.path.abspath(__file__))

JUDGES = {
    "llama":    os.path.join(_REPO, "annotations", "llama_70b"),
    "qwen":     os.path.join(_REPO, "annotations", "qwen_72b"),
    "deepseek": os.path.join(_REPO, "annotations", "deepseek_70b"),
}
CACHE_DIR = os.path.join(_REPO, "eval_cache")
PLOT_DIR  = os.path.join(_REPO, "eval_plots")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ReduceToListOfListOfWords(),
])

# ─── Filename parser ───────────────────────────────────────────────────────────

def parse_filename(fname):
    """Extract model, variant, dataset from filename."""
    base = os.path.basename(fname).replace("_predictions.json", "")
    # Split on -combined or _pretrained
    m = re.match(r"^(.+?)-(combined|pretrained)_(.+)$", base)
    if not m:
        # try with underscore variant separator
        m = re.match(r"^(.+?)_(pretrained|combined)_(.+)$", base)
    if not m:
        return None, None, None
    model   = m.group(1)
    variant = m.group(2)
    dataset = m.group(3)
    return model, variant, dataset

# ─── Metric helpers ────────────────────────────────────────────────────────────

def compute_corpus_wer(references, hypotheses):
    # Apply transforms individually to detect empties post-transform
    tr = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ExpandCommonEnglishContractions(),
    ])
    filtered = [(r, h) for r, h in zip(references, hypotheses)
                if tr([r])[0].strip() and tr([h])[0].strip()]
    refs, hyps = zip(*filtered)
    return round(jiwer.wer(
        list(refs), list(hyps),
        reference_transform=WER_TRANSFORM,
        hypothesis_transform=WER_TRANSFORM,
    ) * 100, 4)


def compute_bertscore(references, hypotheses):
    # Filter empty pairs
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip() and h.strip()]
    refs, hyps = zip(*pairs)
    _, _, F1 = bert_score_fn(
        list(hyps), list(refs),
        lang="en",
        rescale_with_baseline=False,
        model_type="roberta-large",
        batch_size=32,
        device=DEVICE,
        verbose=False,
    )
    return round(1.0 - F1.mean().item(), 6)  # BERTDist: lower is better


class SemDist:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print(f"[SemDist] Loading roberta-large on {DEVICE}...")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.model = RobertaModel.from_pretrained("roberta-large").to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def _embed(self, texts, batch_size=32):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=128, return_tensors="pt").to(DEVICE)
            out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_emb.append(emb.cpu())
        return torch.cat(all_emb, 0)

    def compute(self, references, hypotheses):
        # Filter empty pairs
        pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip() and h.strip()]
        refs, hyps = zip(*pairs)
        r = self._embed(list(refs))
        h = self._embed(list(hyps))
        dist = 1.0 - torch.nn.functional.cosine_similarity(r, h, dim=1)
        return round(dist.mean().item(), 6)


def compute_contextual_error_rate(records):
    """CCER = critical / (critical + equivalent), excludes failed JSON parse records."""
    valid      = [r for r in records if r.get("contextual_status") != "Error"]
    critical   = sum(1 for r in valid if r.get("contextual_status") == "Critical_Errors")
    equivalent = sum(1 for r in valid if r.get("contextual_status") == "Equivalent")
    total = critical + equivalent
    return round(critical / total, 6) if total else None

# ─── Per-file evaluation ───────────────────────────────────────────────────────

def evaluate_file(fpath):
    with open(fpath) as f:
        data = json.load(f)
    records = data.get("records", [])
    refs = [r["reference"]  for r in records]
    hyps = [r["hypothesis"] for r in records]
    return {
        "WER":      compute_corpus_wer(refs, hyps),
        "BERTDist": compute_bertscore(refs, hyps),   # renamed
        "SemDist":  SemDist.get().compute(refs, hyps),
        "CCER":     compute_contextual_error_rate(records),  # renamed
    }

# ─── Main loop ─────────────────────────────────────────────────────────────────

def run_judge(judge_name, judge_dir):
    cache_path = os.path.join(CACHE_DIR, f"{judge_name}_results.csv")

    if os.path.exists(cache_path):
        print(f"[{judge_name}] Cache found, loading...")
        return pd.read_csv(cache_path)

    files = glob.glob(os.path.join(judge_dir, "*.json"))
    rows  = []
    for fpath in sorted(files):
        model, variant, dataset = parse_filename(fpath)
        if model is None:
            print(f"  Skipping unrecognized: {fpath}")
            continue
        print(f"  [{judge_name}] {model} | {variant} | {dataset}")
        metrics = evaluate_file(fpath)
        rows.append({"judge": judge_name, "model": model,
                     "variant": variant, "dataset": dataset, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"[{judge_name}] Saved to {cache_path}")
    return df

# ─── Plotting ──────────────────────────────────────────────────────────────────

METRICS = {
    "WER":    ("Corpus WER (%)",         "lower is better"),
    "BERTDist": ("BERTDist (1 - BERTScore F1, roberta-large)", "lower is better"),
    "SemDist":  ("SemDist (roberta-large)",                     "lower is better"),
    "CCER":     ("Contextual Critical Error Rate",              "lower is better"),
}

DATASETS = ["atco2_ood", "uwb_atcc_test", "atcosim_test"]
MODELS   = ["whisper-medium", "parakeet-tdt-0.6b-v3"]
VARIANTS = ["pretrained", "combined"]
COLORS   = {"pretrained": "#4C72B0", "combined": "#DD8452"}


# def plot_metric(df, judge_name, metric, ylabel, note):
#     fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 5), sharey=False)
#     fig.suptitle(f"[{judge_name.upper()}] {ylabel}", fontsize=13, fontweight="bold")

#     for ax, model in zip(axes, MODELS):
#         mdf = df[df["model"] == model]
#         x   = np.arange(len(DATASETS))
#         w   = 0.35

#         for i, variant in enumerate(VARIANTS):
#             vdf    = mdf[mdf["variant"] == variant].set_index("dataset")
#             vals   = [vdf.loc[ds, metric] if ds in vdf.index else np.nan for ds in DATASETS]
#             offset = (i - 0.5) * w
#             bars   = ax.bar(x + offset, vals, w, label=variant,
#                             color=COLORS[variant], edgecolor="white", linewidth=0.5)
#             for bar, val in zip(bars, vals):
#                 if not np.isnan(val):
#                     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
#                             f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

#         ax.set_title(model, fontsize=10)
#         ax.set_xticks(x)
#         ax.set_xticklabels([d.replace("_test","").replace("_"," ") for d in DATASETS],
#                            fontsize=8, rotation=15)
#         ax.set_ylabel(ylabel, fontsize=9)
#         ax.legend(fontsize=8)
#         ax.grid(axis="y", linestyle="--", alpha=0.4)
#         ax.spines[["top","right"]].set_visible(False)

#     fig.text(0.5, 0.01, note, ha="center", fontsize=8, color="gray")
#     plt.tight_layout(rect=[0, 0.04, 1, 1])
#     out = os.path.join(PLOT_DIR, f"{judge_name}_{metric}.png")
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {out}")


# def plot_all(df, judge_name):
#     for metric, (ylabel, note) in METRICS.items():
#         plot_metric(df, judge_name, metric, ylabel, note)

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_dfs = []
    for judge_name, judge_dir in JUDGES.items():
        df = run_judge(judge_name, judge_dir)
        plot_all(df, judge_name)
        all_dfs.append(df)

    combined = pd.concat(all_dfs)
    combined.to_csv(os.path.join(CACHE_DIR, "all_results.csv"), index=False)
    print("\nDone. All plots saved to:", PLOT_DIR)