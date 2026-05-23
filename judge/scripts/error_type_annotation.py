"""
Error Type Annotation Dataset Preparation
==========================================
Prepares a dataset for human annotation of ATC error categories.
Outputs both a wide CSV (for κ computation) and a long CSV (for human annotators).
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# ============================================================
# CONFIG — update paths to match your environment
# ============================================================

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JUDGES = {
    "llama":    os.path.join(_REPO, "annotations", "llama_70b"),
    "qwen":     os.path.join(_REPO, "annotations", "qwen_72b"),
    "deepseek": os.path.join(_REPO, "annotations", "deepseek_70b"),
}

ANNOTATION_SHEET  = os.path.join(_REPO, "eval_cache", "annotation_sheet_shared.csv")
OUT_WIDE          = os.path.join(_REPO, "eval_cache", "error_type_annotation_wide.csv")
OUT_LONG          = os.path.join(_REPO, "eval_cache", "error_type_annotation_long.csv")

VALID_CLASSES = [
    "callsign", "altitude", "heading", "runway",
    "frequency", "clearance", "navigation", "weather", "other"
]
VALID_CLASSES_SET = set(VALID_CLASSES)

# ============================================================
# STEP 1 — Frequency analysis across all critical records
# ============================================================

def get_error_types(records):
    """
    For each Critical_Errors record, return the set of valid entity_types.
    Multi-label: a record can have multiple types.
    """
    out = []
    for r in records:
        if r.get("contextual_status") != "Critical_Errors":
            continue
        types = set()
        for err in r.get("transcription_errors", []):
            et = err.get("entity_type", "").lower().strip()
            if et in VALID_CLASSES_SET:
                types.add(et)
        if types:
            out.append(types)
    return out


print("=" * 60)
print("STEP 1: Error type frequency per judge")
print("=" * 60)

for judge_name, judge_dir in JUDGES.items():
    all_types = []
    for fpath in sorted(glob.glob(os.path.join(judge_dir, "*.json"))):
        with open(fpath) as f:
            records = json.load(f)["records"]
        all_types.extend(get_error_types(records))

    if not all_types:
        print(f"\n[{judge_name}] No critical records found — check path: {judge_dir}")
        continue

    class_counts = Counter()
    for types in all_types:
        for t in types:
            class_counts[t] += 1

    print(f"\n[{judge_name}] Critical records with ≥1 error type: {len(all_types)}")
    print(f"  Multi-label (>1 type): {sum(1 for t in all_types if len(t) > 1)}")
    for cls, cnt in class_counts.most_common():
        print(f"  {cls:15s}: {cnt:5d}  ({100 * cnt / len(all_types):.1f}%)")

# ============================================================
# STEP 2 — Build composite key lookup per judge
# ============================================================

def load_judge_index(judge_dir):
    """
    Returns dict: (reference, hypothesis) -> frozenset of entity_types

    Uses a composite key to avoid collisions when the same reference
    appears with different hypotheses across datasets.
    Last-write-wins within a single (ref, hyp) pair, which is safe
    since a given ASR output is deterministic per model.
    """
    index = {}
    for fpath in sorted(glob.glob(os.path.join(judge_dir, "*.json"))):
        with open(fpath) as f:
            records = json.load(f)["records"]
        for r in records:
            if r.get("contextual_status") != "Critical_Errors":
                continue
            key = (
                r.get("reference",   "").strip().lower(),
                r.get("hypothesis",  "").strip().lower(),  # adjust field name if different
            )
            types = set()
            for err in r.get("transcription_errors", []):
                et = err.get("entity_type", "").lower().strip()
                if et in VALID_CLASSES_SET:
                    types.add(et)
            index[key] = frozenset(types)
    return index


print("\n" + "=" * 60)
print("STEP 2: Building judge indexes")
print("=" * 60)

judge_indexes = {}
for judge_name, judge_dir in JUDGES.items():
    idx = load_judge_index(judge_dir)
    judge_indexes[judge_name] = idx
    print(f"  [{judge_name}] Indexed {len(idx)} critical (ref, hyp) pairs")

# ============================================================
# STEP 3 — Load annotation sheet and filter to agreed_critical
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Loading annotation sheet")
print("=" * 60)

df = pd.read_csv(ANNOTATION_SHEET)
critical_df = df[df["sample_type"] == "agreed_critical"].copy().reset_index(drop=True)
print(f"  Agreed-critical samples: {len(critical_df)}")

# Normalised key columns for joining
critical_df["_ref_key"] = critical_df["reference"].str.strip().str.lower()
critical_df["_hyp_key"] = critical_df["hypothesis"].str.strip().str.lower()

# ============================================================
# STEP 4 — Add per-class binary columns for each judge
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Adding judge binary columns")
print("=" * 60)

# Track how many samples each judge matched
match_counts = defaultdict(int)

for cls in VALID_CLASSES:
    for judge_name, idx in judge_indexes.items():
        col = f"judge_{judge_name}_{cls}"

        def lookup(row, _idx=idx, _cls=cls):
            key = (row["_ref_key"], row["_hyp_key"])
            types = _idx.get(key)
            if types is None:
                return np.nan       # judge never saw this sample
            return int(_cls in types)

        critical_df[col] = critical_df.apply(lookup, axis=1)

# Report match rate per judge (using first class as proxy — same for all)
for judge_name in judge_indexes:
    col = f"judge_{judge_name}_{VALID_CLASSES[0]}"
    matched = critical_df[col].notna().sum()
    print(f"  [{judge_name}] Matched {matched}/{len(critical_df)} samples")
    if matched < len(critical_df):
        missing = critical_df[critical_df[col].isna()][["reference", "hypothesis"]].head(5)
        print(f"    First unmatched samples:")
        print(missing.to_string(index=False))

# Human annotation placeholders — NaN so downstream κ can use dropna()
for cls in VALID_CLASSES:
    critical_df[f"human1_{cls}"] = np.nan
    critical_df[f"human2_{cls}"] = np.nan

# Drop internal key columns
critical_df.drop(columns=["_ref_key", "_hyp_key"], inplace=True)

# ============================================================
# STEP 5 — Save wide CSV (one row per sample, all binary columns)
# ============================================================

os.makedirs(os.path.dirname(OUT_WIDE), exist_ok=True)
critical_df.to_csv(OUT_WIDE, index=False)
print(f"\n[Wide CSV] Saved {OUT_WIDE}")
print(f"  Shape: {critical_df.shape}")
print(f"  Columns: {list(critical_df.columns)}")

# ============================================================
# STEP 6 — Save long CSV (one row per sample × class — easier to annotate)
#
# Each human annotator fills in a single 0/1 column (human1 or human2)
# per row. Judge votes are shown for context.
# ============================================================

rows = []
for _, row in critical_df.iterrows():
    for cls in VALID_CLASSES:
        rows.append({
            "reference":        row["reference"],
            "hypothesis":       row.get("hypothesis", ""),
            "error_class":      cls,
            # Judge votes (may be NaN if judge didn't see this sample)
            "judge_llama":      row.get(f"judge_llama_{cls}",    np.nan),
            "judge_qwen":       row.get(f"judge_qwen_{cls}",     np.nan),
            "judge_deepseek":   row.get(f"judge_deepseek_{cls}", np.nan),
            # Human annotators fill these in (0 = not this class, 1 = yes this class)
            "human1":           np.nan,
            "human2":           np.nan,
        })

long_df = pd.DataFrame(rows)
long_df.to_csv(OUT_LONG, index=False)
print(f"\n[Long CSV] Saved {OUT_LONG}")
print(f"  Shape: {long_df.shape}  ({len(critical_df)} samples × {len(VALID_CLASSES)} classes)")

# ============================================================
# STEP 7 — Sanity checks
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Sanity checks")
print("=" * 60)

# Check inter-judge agreement on class assignments (before human annotation)
judge_names = list(judge_indexes.keys())
if len(judge_names) >= 2:
    from sklearn.metrics import cohen_kappa_score

    print("\n  Pairwise inter-judge Cohen's κ per class:")
    print(f"  {'class':15s}", end="")
    pairs = [(judge_names[i], judge_names[j])
             for i in range(len(judge_names))
             for j in range(i+1, len(judge_names))]
    for a, b in pairs:
        print(f"  {a[:5]} vs {b[:5]}", end="")
    print()

    for cls in VALID_CLASSES:
        print(f"  {cls:15s}", end="")
        for a, b in pairs:
            col_a = f"judge_{a}_{cls}"
            col_b = f"judge_{b}_{cls}"
            subset = critical_df[[col_a, col_b]].dropna()
            if len(subset) < 2 or subset[col_a].nunique() < 2 or subset[col_b].nunique() < 2:
                print(f"  {'N/A':>12s}", end="")
            else:
                kappa = cohen_kappa_score(subset[col_a].astype(int), subset[col_b].astype(int))
                print(f"  {kappa:>12.3f}", end="")
        print()

print("\nDone. Share OUT_LONG with human annotators.")
print("After annotation, use OUT_WIDE for κ computation across all annotators.")