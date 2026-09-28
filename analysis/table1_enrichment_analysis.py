"""
Corrected Table 1: Finetuning Paradox, all four metrics.

Fixes relative to the original table1_finetuning_paradox.py:
  (1) Labels are the 3-judge majority vote (llama_70b, qwen_72b, deepseek_70b)
      -- the validated CxER ensemble reported everywhere else in the paper --
      not llama_70b alone.
  (2) Reports THREE quantities per (metric, model, dataset), not one:
        - P(low | critical): fraction of critical-error samples whose metric
          value is below tau (the original script's metric -- concentration
          of remaining failures in the "looks safe" region).
        - coverage: P(metric < tau) over ALL samples -- how much the "safe"
          region itself grows after fine-tuning. Reported so the reader can
          see the denominator shift, not just the raw delta.
        - enrichment: P(low|critical) / coverage -- population-shift-
          controlled version. Are critical errors over/under-represented in
          the safe region relative to base rate, not just in absolute terms.
  tau (per metric) is the 25th percentile of pretrained values, pooled across
  both models and all three datasets, computed over majority-vote-valid
  pretrained records -- same convention as table1_bootstrap_ci.py.

  Supports the "Ruling out a denominator artifact" paragraph (Sec. 2.4):
  denominator (coverage) growth alone would keep the enrichment ratio flat;
  instead it rises in 23 of 24 (metric, model, dataset) cells.
"""

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

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOT_ROOT = os.path.join(_REPO, "annotations")
OUT_DIR = os.path.join(_REPO, "analysis_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

JUDGES = ["llama_70b", "qwen_72b", "deepseek_70b"]
MODELS = ["whisper-medium", "parakeet-tdt-0.6b-v3"]
DATASETS = ["atco2_ood", "uwb_atcc_test", "atcosim_test"]
METRICS = ["WER", "WeightedWER", "BERTDist", "SemDist"]

MODEL_LABEL = {"whisper-medium": "Whisper", "parakeet-tdt-0.6b-v3": "Parakeet"}
DATASET_LABEL = {"atco2_ood": "ATCO2", "uwb_atcc_test": "ATCC", "atcosim_test": "ATCOSim"}


def fname(model, variant, dataset):
    return f"{model}_pretrained_{dataset}_predictions.json" if variant == "pretrained" \
        else f"{model}-combined_{dataset}_predictions.json"


def load_judge(judge, model, variant, dataset):
    with open(os.path.join(ANNOT_ROOT, judge, fname(model, variant, dataset))) as f:
        return json.load(f)["records"]


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
    def per_record(self, refs, hyps, batch_size=64):
        def embed(texts):
            out_emb = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True,
                                      max_length=128, return_tensors="pt").to(DEVICE)
                out = self.model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                out_emb.append(emb.cpu())
            return torch.cat(out_emb, 0)
        r, h = embed(refs), embed(hyps)
        return (1.0 - torch.nn.functional.cosine_similarity(r, h, dim=1)).numpy()


def bertdist_per_record(refs, hyps):
    _, _, F1 = bert_score_fn(hyps, refs, lang="en", rescale_with_baseline=False,
                              model_type="roberta-large", batch_size=64, device=DEVICE, verbose=False)
    return (1.0 - F1.numpy())


print("Loading records, majority-vote labels, and per-record metrics (WER, WeightedWER, BERTDist, SemDist)...")
data = {}
for model in MODELS:
    for variant in ["pretrained", "combined"]:
        for ds in DATASETS:
            per_judge = {j: load_judge(j, model, variant, ds) for j in JUDGES}
            n = len(per_judge["llama_70b"])
            recs = []
            for i in range(n):
                wer = per_judge["llama_70b"][i].get("WER")
                if wer is None:
                    continue
                votes = [per_judge[j][i]["contextual_status"] for j in JUDGES
                         if per_judge[j][i]["contextual_status"] in ("Critical_Errors", "Equivalent")]
                if not votes:
                    continue
                label = "Critical_Errors" if votes.count("Critical_Errors") * 2 > len(votes) else "Equivalent"
                ref = per_judge["llama_70b"][i]["reference"]
                hyp = per_judge["llama_70b"][i]["hypothesis"]
                recs.append({"WER": wer, "label": label, "reference": ref, "hypothesis": hyp})

            refs = [r["reference"] for r in recs]
            hyps = [r["hypothesis"] for r in recs]
            bd = bertdist_per_record(refs, hyps)
            sd = SemDist.get().per_record(refs, hyps)
            for i, r in enumerate(recs):
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

            data[(model, variant, ds)] = recs
            print(f"  {MODEL_LABEL[model]} {variant} {ds}: {len(recs)} records")

# ── thresholds: 25th pct of pooled pretrained values per metric ─────────────
thresholds = {}
for metric in METRICS:
    vals = []
    for model in MODELS:
        for ds in DATASETS:
            vals.extend([r[metric] for r in data[(model, "pretrained", ds)]])
    thresholds[metric] = float(np.percentile(vals, 25))
    print(f"tau[{metric}] = {thresholds[metric]:.4f}")


def summarize(records, metric, tau):
    n = len(records)
    below = [r for r in records if r[metric] < tau]
    crit = [r for r in records if r["label"] == "Critical_Errors"]
    crit_below = [r for r in crit if r[metric] < tau]
    coverage = len(below) / n
    p_low_given_crit = len(crit_below) / len(crit) if crit else float("nan")
    enrichment = p_low_given_crit / coverage if coverage else float("nan")
    return coverage, p_low_given_crit, enrichment


rows = []
for metric in METRICS:
    tau = thresholds[metric]
    for model in MODELS:
        for ds in DATASETS:
            cov_pre, low_pre, enr_pre = summarize(data[(model, "pretrained", ds)], metric, tau)
            cov_post, low_post, enr_post = summarize(data[(model, "combined", ds)], metric, tau)
            rows.append({
                "Metric": metric, "Model": MODEL_LABEL[model], "Dataset": DATASET_LABEL[ds],
                "coverage_pre": round(100 * cov_pre, 1), "coverage_post": round(100 * cov_post, 1),
                "P(low|crit)_pre": round(100 * low_pre, 1), "P(low|crit)_post": round(100 * low_post, 1),
                "P(low|crit)_delta": round(100 * (low_post - low_pre), 1),
                "enrichment_pre": round(enr_pre, 3), "enrichment_post": round(enr_post, 3),
                "enrichment_delta": round(enr_post - enr_pre, 3),
            })

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
print("\n" + df.to_string(index=False))

out_path = os.path.join(OUT_DIR, "table1_corrected_full.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
