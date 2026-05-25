import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score_fn
from transformers import RobertaTokenizer, RobertaModel
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weighted_wer import tokenise, entity_token_mask, _levenshtein_weighted, ENTITY_WEIGHT

_REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(_REPO, "annotations", "llama_70b")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = ["atco2_ood", "uwb_atcc_test", "atcosim_test"]
MODELS   = ["whisper-medium", "parakeet-tdt-0.6b-v3"]

MODEL_LABEL = {
    "whisper-medium":        "Whisper",
    "parakeet-tdt-0.6b-v3": "Parakeet",
}
DATASET_LABEL = {
    "atco2_ood":     "ATCO2",
    "uwb_atcc_test": "ATCC",
    "atcosim_test":  "ATCOSim",
}

# Filename pattern: pretrained uses underscore, combined uses hyphen
def fname(model, variant, dataset):
    if variant == "pretrained":
        return f"{model}_pretrained_{dataset}_predictions.json"
    else:
        return f"{model}-combined_{dataset}_predictions.json"


def load(model, variant, dataset):
    path = os.path.join(BASE_DIR, fname(model, variant, dataset))
    with open(path) as f:
        return json.load(f)["records"]


def get_valid(records):
    return [
        r for r in records
        if r["contextual_status"] in ("Critical_Errors", "Equivalent")
        and r.get("WER") is not None
    ]


# ── SemDist (roberta-large cosine distance per record) ───────────────────────

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
        self.model     = RobertaModel.from_pretrained("roberta-large").to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def _embed(self, texts, batch_size=32):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc   = self.tokenizer(batch, padding=True, truncation=True,
                                   max_length=128, return_tensors="pt").to(DEVICE)
            out   = self.model(**enc)
            mask  = enc["attention_mask"].unsqueeze(-1).float()
            emb   = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_emb.append(emb.cpu())
        return torch.cat(all_emb, 0)

    def per_record(self, refs, hyps):
        r    = self._embed(refs)
        h    = self._embed(hyps)
        dist = 1.0 - torch.nn.functional.cosine_similarity(r, h, dim=1)
        return dist.numpy()


def bertscore_per_record(refs, hyps):
    _, _, F1 = bert_score_fn(
        hyps, refs,
        lang="en",
        rescale_with_baseline=False,
        model_type="roberta-large",
        batch_size=32,
        device=DEVICE,
        verbose=False,
    )
    return (1.0 - F1.numpy())   # BERTDist: lower is better


# ── Load all records and attach per-record metric values ─────────────────────

print("Loading and scoring all records (this may take a few minutes on CPU)...")

data = {}   # (model, variant, dataset) → list of records with metric fields

for model in MODELS:
    for variant in ["pretrained", "combined"]:
        for ds in DATASETS:
            records = get_valid(load(model, variant, ds))
            refs    = [r["reference"]  for r in records]
            hyps    = [r["hypothesis"] for r in records]

            bd  = bertscore_per_record(refs, hyps)
            sd  = SemDist.get().per_record(refs, hyps)

            for i, r in enumerate(records):
                r["BERTDist"] = float(bd[i])
                r["SemDist"]  = float(sd[i])
                # per-record WeightedWER (normalised to [0,1] like WER)
                ref_tok  = tokenise(r["reference"])
                hyp_tok  = tokenise(r["hypothesis"])
                ref_mask = entity_token_mask(ref_tok)
                if ref_tok:
                    err, cost    = _levenshtein_weighted(ref_tok, hyp_tok, ref_mask, ENTITY_WEIGHT)
                    r["WeightedWER"] = err / cost
                else:
                    r["WeightedWER"] = 0.0

            data[(model, variant, ds)] = records
            print(f"  {MODEL_LABEL[model]} {variant} {ds}: {len(records)} records")

# ── Compute thresholds (25th percentile of pretrained records) ────────────────

print("\nComputing thresholds...")

metrics = ["WER", "WeightedWER", "BERTDist", "SemDist"]
thresholds = {}

for metric in metrics:
    all_vals = []
    for model in MODELS:
        for ds in DATASETS:
            recs = data[(model, "pretrained", ds)]
            all_vals.extend([r[metric] for r in recs])
    thresholds[metric] = np.percentile(all_vals, 25)
    print(f"  {metric} 25th-pct threshold: {thresholds[metric]:.6f}")

# ── Compute % critical in low-metric region, then delta ──────────────────────

def pct_critical_in_low_region(records, metric, threshold):
    critical = [r for r in records if r["contextual_status"] == "Critical_Errors"]
    if not critical:
        return np.nan
    low = [r for r in critical if r[metric] < threshold]
    return 100.0 * len(low) / len(critical)


print("\n")
rows = []
for metric in metrics:
    thr = thresholds[metric]
    for model in MODELS:
        row = {"Metric": f"Δ_{metric}", "Model": MODEL_LABEL[model]}
        for ds in DATASETS:
            pre  = pct_critical_in_low_region(data[(model, "pretrained", ds)], metric, thr)
            post = pct_critical_in_low_region(data[(model, "combined",   ds)], metric, thr)
            delta = post - pre
            row[DATASET_LABEL[ds]] = round(delta, 1)
        rows.append(row)

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# ── LaTeX table ───────────────────────────────────────────────────────────────

print("\n=== LaTeX ===")
print(r"\begin{tabular}{@{}llccc@{}}")
print(r"\toprule")
print(r"\textbf{Metric} & \textbf{Model} & \textbf{ATCO2} & \textbf{ATCC} & \textbf{ATCOSim} \\")
print(r"\midrule")
for i, row in df.iterrows():
    metric = row["Metric"]
    model  = row["Model"]
    vals   = f"{row['ATCO2']:+.1f} & {row['ATCC']:+.1f} & {row['ATCOSim']:+.1f}"
    if model == "Whisper":
        print(rf"\multirow{{2}}{{*}}{{${metric}$}}")
    print(rf"& {model} & {vals} \\")
    if model == "Parakeet" and i < len(df) - 1:
        print(r"\midrule")
print(r"\bottomrule")
print(r"\end{tabular}")
