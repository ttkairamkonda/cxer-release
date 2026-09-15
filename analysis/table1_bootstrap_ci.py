"""
95% bootstrap CIs for Table 1 (the Finetuning Paradox) Delta values.

Delta = pct_critical_in_low_region(fine-tuned) - pct_critical_in_low_region(pretrained)

This reuses table1_finetuning_paradox.py's exact per-record metric
computation and global-threshold logic (so tau and the underlying values are
identical to the reported Table 1), then bootstrap-resamples the
Critical_Errors subset within each (metric, model, dataset, variant) cell
1000 times to get a 95% CI on Delta. Per-record metric computation
(BERTScore/SemDist) is the expensive part and is done exactly once; the
bootstrap itself is pure resampling over already-computed per-record values,
so it's fast.
"""

import os
import json
import numpy as np
import torch
from bert_score import score as bert_score_fn
from transformers import RobertaTokenizer, RobertaModel
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weighted_wer import tokenise, entity_token_mask, _levenshtein_weighted, ENTITY_WEIGHT

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(_REPO, "annotations", "llama_70b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = ["atco2_ood", "uwb_atcc_test", "atcosim_test"]
MODELS = ["whisper-medium", "parakeet-tdt-0.6b-v3"]
MODEL_LABEL = {"whisper-medium": "Whisper", "parakeet-tdt-0.6b-v3": "Parakeet"}
DATASET_LABEL = {"atco2_ood": "ATCO2", "uwb_atcc_test": "ATCC", "atcosim_test": "ATCOSim"}

N_BOOTSTRAP = 1000
rng = np.random.default_rng(42)


def fname(model, variant, dataset):
    return f"{model}_pretrained_{dataset}_predictions.json" if variant == "pretrained" \
        else f"{model}-combined_{dataset}_predictions.json"


def load(model, variant, dataset):
    with open(os.path.join(BASE_DIR, fname(model, variant, dataset))) as f:
        return json.load(f)["records"]


def get_valid(records):
    return [r for r in records if r["contextual_status"] in ("Critical_Errors", "Equivalent")
            and r.get("WER") is not None]


class SemDist:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print(f"[SemDist] Loading roberta-large on {DEVICE}...", flush=True)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.model = RobertaModel.from_pretrained("roberta-large").to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def _embed(self, texts, batch_size=32):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                  max_length=128, return_tensors="pt").to(DEVICE)
            out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_emb.append(emb.cpu())
        return torch.cat(all_emb, 0)

    def per_record(self, refs, hyps):
        r = self._embed(refs)
        h = self._embed(hyps)
        dist = 1.0 - torch.nn.functional.cosine_similarity(r, h, dim=1)
        return dist.numpy()


def bertscore_per_record(refs, hyps):
    _, _, F1 = bert_score_fn(
        hyps, refs, lang="en", rescale_with_baseline=False,
        model_type="roberta-large", batch_size=32, device=DEVICE, verbose=False,
    )
    return (1.0 - F1.numpy())


print("Loading and scoring all records (expensive step, done once)...", flush=True)

data = {}
for model in MODELS:
    for variant in ["pretrained", "combined"]:
        for ds in DATASETS:
            records = get_valid(load(model, variant, ds))
            refs = [r["reference"] for r in records]
            hyps = [r["hypothesis"] for r in records]
            bd = bertscore_per_record(refs, hyps)
            sd = SemDist.get().per_record(refs, hyps)
            for i, r in enumerate(records):
                r["BERTDist"] = float(bd[i])
                r["SemDist"] = float(sd[i])
                ref_tok = tokenise(r["reference"])
                hyp_tok = tokenise(r["hypothesis"])
                ref_mask = entity_token_mask(ref_tok)
                if ref_tok:
                    err, cost = _levenshtein_weighted(ref_tok, hyp_tok, ref_mask, ENTITY_WEIGHT)
                    r["WeightedWER"] = err / cost
                else:
                    r["WeightedWER"] = 0.0
            data[(model, variant, ds)] = records
            print(f"  {MODEL_LABEL[model]} {variant} {ds}: {len(records)} records", flush=True)

metrics = ["WER", "WeightedWER", "BERTDist", "SemDist"]
thresholds = {}
for metric in metrics:
    all_vals = []
    for model in MODELS:
        for ds in DATASETS:
            all_vals.extend([r[metric] for r in data[(model, "pretrained", ds)]])
    thresholds[metric] = np.percentile(all_vals, 25)
    print(f"  {metric} tau = {thresholds[metric]:.6f}", flush=True)


def pct_critical_in_low_region(records, metric, threshold):
    critical = [r for r in records if r["contextual_status"] == "Critical_Errors"]
    if not critical:
        return np.nan
    low = sum(1 for r in critical if r[metric] < threshold)
    return 100.0 * low / len(critical)


def bootstrap_delta(records_pre, records_ft, metric, threshold, n=N_BOOTSTRAP):
    """
    Bootstrap resample the Critical_Errors subset of each variant
    (fixed tau, as estimated from the real data) and recompute Delta
    = pct_low(ft) - pct_low(pre) each replicate.
    """
    crit_pre = [r for r in records_pre if r["contextual_status"] == "Critical_Errors"]
    crit_ft = [r for r in records_ft if r["contextual_status"] == "Critical_Errors"]
    if not crit_pre or not crit_ft:
        return (np.nan, np.nan)

    low_pre = np.array([1 if r[metric] < threshold else 0 for r in crit_pre])
    low_ft = np.array([1 if r[metric] < threshold else 0 for r in crit_ft])

    n_pre, n_ft = len(low_pre), len(low_ft)
    deltas = np.empty(n)
    for b in range(n):
        idx_pre = rng.integers(0, n_pre, n_pre)
        idx_ft = rng.integers(0, n_ft, n_ft)
        pct_pre = 100.0 * low_pre[idx_pre].mean()
        pct_ft = 100.0 * low_ft[idx_ft].mean()
        deltas[b] = pct_ft - pct_pre
    return tuple(np.percentile(deltas, [2.5, 97.5]))


print("\n=== TABLE 1 DELTA 95% BOOTSTRAP CIs (tau fixed at estimated value, 1000 resamples) ===", flush=True)
rows = []
for metric in metrics:
    thr = thresholds[metric]
    for model in MODELS:
        for ds in DATASETS:
            pre = data[(model, "pretrained", ds)]
            ft = data[(model, "combined", ds)]
            point_pre = pct_critical_in_low_region(pre, metric, thr)
            point_ft = pct_critical_in_low_region(ft, metric, thr)
            delta = point_ft - point_pre
            ci_lo, ci_hi = bootstrap_delta(pre, ft, metric, thr)
            row = {
                "metric": metric, "model": MODEL_LABEL[model], "dataset": DATASET_LABEL[ds],
                "delta": round(delta, 1), "ci_lo": round(ci_lo, 1), "ci_hi": round(ci_hi, 1),
            }
            rows.append(row)
            print(f"  D_{metric:12s} {MODEL_LABEL[model]:9s} {DATASET_LABEL[ds]:8s}  "
                  f"Delta={delta:+.1f}  95% CI [{ci_lo:+.1f}, {ci_hi:+.1f}]", flush=True)

import csv
out_csv = os.path.join(_REPO, "prompt_robustness_analysis", "table1_bootstrap_ci.csv")
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["metric", "model", "dataset", "delta", "ci_lo", "ci_hi"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved -> {out_csv}")
