"""
Error Type Annotation Dataset Preparation + Evaluation
======================================================

This script:

1. Builds annotation sheets from judge JSON outputs
2. Saves:
      - Wide CSV
      - Long CSV
3. AFTER human annotation:
      - Loads annotated long CSV
      - Merges human labels back into wide format
      - Computes:
            * Cohen's κ
            * Krippendorff's α (per class and pooled)
            * Precision / Recall / F1 / Accuracy
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import krippendorff

from collections import Counter, defaultdict
from sklearn.metrics import (
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JUDGES = {
    "llama":    os.path.join(_REPO, "annotations", "llama_70b"),
    "qwen":     os.path.join(_REPO, "annotations", "qwen_72b"),
    "deepseek": os.path.join(_REPO, "annotations", "deepseek_70b"),
}

ANNOTATION_SHEET = os.path.join(_REPO, "eval_cache", "annotation_sheet_shared.csv")
OUT_WIDE         = os.path.join(_REPO, "eval_cache", "error_type_annotation_wide.csv")
OUT_LONG         = os.path.join(_REPO, "eval_cache", "error_type_annotation_long.csv")
ANNOTATED_LONG   = os.path.join(_REPO, "eval_cache", "error_type_annotation_long_annotated.csv")

VALID_CLASSES = [
    "callsign", "altitude", "heading", "runway",
    "frequency", "clearance", "navigation", "weather", "other",
]
VALID_CLASSES_SET = set(VALID_CLASSES)


def interpret(k):
    if k < 0:   return "Less than chance"
    if k < 0.2: return "Slight"
    if k < 0.4: return "Fair"
    if k < 0.6: return "Moderate"
    if k < 0.8: return "Substantial"
    return "Almost perfect"


# ── Step 1: frequency analysis ───────────────────────────────────────────────

def get_error_types(records):
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
        print(f"\n[{judge_name}] No critical records found")
        continue

    class_counts = Counter()
    for types in all_types:
        for t in types:
            class_counts[t] += 1

    print(f"\n[{judge_name}] Critical records with ≥1 type: {len(all_types)}")
    print(f"  Multi-label (>1 type): {sum(1 for t in all_types if len(t) > 1)}")
    for cls, cnt in class_counts.most_common():
        print(f"  {cls:15s}: {cnt:5d} ({100 * cnt / len(all_types):.1f}%)")


# ── Step 2: build judge indexes ───────────────────────────────────────────────

def load_judge_index(judge_dir):
    index = {}
    for fpath in sorted(glob.glob(os.path.join(judge_dir, "*.json"))):
        with open(fpath) as f:
            data = json.load(f)
        for r in data["records"]:
            if r.get("contextual_status") != "Critical_Errors":
                continue
            key = (
                r.get("reference", "").strip().lower(),
                r.get("hypothesis", "").strip().lower(),
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
    print(f"  [{judge_name}] Indexed {len(idx)} critical pairs")


# ── Step 3: load annotation sheet ────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Loading annotation sheet")
print("=" * 60)

df = pd.read_csv(ANNOTATION_SHEET)
critical_df = df[df["sample_type"] == "agreed_critical"].copy().reset_index(drop=True)
print(f"  Agreed-critical samples: {len(critical_df)}")

critical_df["_ref_key"] = critical_df["reference"].str.strip().str.lower()
critical_df["_hyp_key"] = critical_df["hypothesis"].str.strip().str.lower()


# ── Step 4: add judge binary columns ─────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Adding judge binary columns")
print("=" * 60)

for cls in VALID_CLASSES:
    for judge_name, idx in judge_indexes.items():
        col = f"judge_{judge_name}_{cls}"

        def lookup(row, _idx=idx, _cls=cls):
            key   = (row["_ref_key"], row["_hyp_key"])
            types = _idx.get(key)
            if types is None:
                return np.nan
            return int(_cls in types)

        critical_df[col] = critical_df.apply(lookup, axis=1)

for judge_name in judge_indexes:
    col     = f"judge_{judge_name}_{VALID_CLASSES[0]}"
    matched = critical_df[col].notna().sum()
    print(f"  [{judge_name}] Matched {matched}/{len(critical_df)} samples")

for cls in VALID_CLASSES:
    critical_df[f"human1_{cls}"] = np.nan

critical_df.drop(columns=["_ref_key", "_hyp_key"], inplace=True)


# ── Step 5 & 6: save wide and long CSVs ──────────────────────────────────────

os.makedirs(os.path.dirname(OUT_WIDE), exist_ok=True)
critical_df.to_csv(OUT_WIDE, index=False)
print(f"\n[Wide CSV] Saved {OUT_WIDE}")
print(f"  Shape: {critical_df.shape}")

rows = []
for _, row in critical_df.iterrows():
    for cls in VALID_CLASSES:
        rows.append({
            "reference":    row["reference"],
            "hypothesis":   row.get("hypothesis", ""),
            "error_class":  cls,
            "judge_llama":    row.get(f"judge_llama_{cls}",    np.nan),
            "judge_qwen":     row.get(f"judge_qwen_{cls}",     np.nan),
            "judge_deepseek": row.get(f"judge_deepseek_{cls}", np.nan),
            "human1":       np.nan,
        })

long_df = pd.DataFrame(rows)
long_df.to_csv(OUT_LONG, index=False)
print(f"\n[Long CSV] Saved {OUT_LONG}")
print(f"  Shape: {long_df.shape}")


# ── Step 7: pairwise inter-judge κ ────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7: Pairwise inter-judge Cohen κ")
print("=" * 60)

judge_names = list(judge_indexes.keys())
pairs = [
    (judge_names[i], judge_names[j])
    for i in range(len(judge_names))
    for j in range(i + 1, len(judge_names))
]

print(f"\n{'class':15s}", end="")
for a, b in pairs:
    print(f"{a[:5]} vs {b[:5]:>12s}", end="")
print()

for cls in VALID_CLASSES:
    print(f"{cls:15s}", end="")
    for a, b in pairs:
        col_a  = f"judge_{a}_{cls}"
        col_b  = f"judge_{b}_{cls}"
        subset = critical_df[[col_a, col_b]].dropna()
        if len(subset) < 2 or subset[col_a].nunique() < 2 or subset[col_b].nunique() < 2:
            print(f"{'N/A':>17s}", end="")
            continue
        kappa = cohen_kappa_score(subset[col_a].astype(int), subset[col_b].astype(int))
        print(f"{kappa:17.3f}", end="")
    print()


# ── Krippendorff's α helpers ─────────────────────────────────────────────────

def krippendorff_alpha_per_class(dataframe, judge_cols, valid_classes):
    """Computes Krippendorff's alpha for each error class across all judges."""
    print("\n=== KRIPPENDORFF'S α PER CLASS ===")
    print(f"  {'class':15s}  {'α':>8s}  interpretation")
    print(f"  {'-'*15}  {'-'*8}  {'-'*20}")

    results = {}
    for cls in valid_classes:
        cols   = [f"judge_{j}_{cls}" for j in judge_cols]
        matrix = dataframe[cols].T.values.astype(float)

        try:
            alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal")
        except Exception as e:
            alpha = float("nan")
            print(f"  {cls:15s}  ERROR: {e}")
            continue

        results[cls] = alpha
        print(f"  {cls:15s}  {alpha:>8.3f}  {interpret(alpha)}")

    return results


def krippendorff_alpha_overall(dataframe, judge_cols, valid_classes):
    """Pools all class columns and computes one overall Krippendorff alpha."""
    all_cols_per_judge = {
        j: [f"judge_{j}_{cls}" for cls in valid_classes]
        for j in judge_cols
    }

    rows = [
        np.concatenate([dataframe[col].values.astype(float) for col in all_cols_per_judge[j]])
        for j in judge_cols
    ]
    matrix = np.array(rows)

    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal")
    print(f"\n  Overall α (all classes pooled): {alpha:.3f}  ({interpret(alpha)})")
    return alpha


# ── Positive count summary ────────────────────────────────────────────────────

print("\nPositive counts per judge per class:")
print(f"  {'class':15s}  {'llama':>8s}  {'qwen':>8s}  {'deepseek':>8s}")
for cls in VALID_CLASSES:
    counts = [
        (critical_df[f"judge_{j}_{cls}"] == 1).sum()
        for j in ["llama", "qwen", "deepseek"]
    ]
    print(f"  {cls:15s}  {counts[0]:>8d}  {counts[1]:>8d}  {counts[2]:>8d}")

alpha_results = krippendorff_alpha_per_class(
    critical_df,
    judge_cols=["llama", "qwen", "deepseek"],
    valid_classes=VALID_CLASSES,
)

alpha_overall = krippendorff_alpha_overall(
    critical_df,
    judge_cols=["llama", "qwen", "deepseek"],
    valid_classes=VALID_CLASSES,
)


# ── Step 8: human evaluation (runs only if annotated file exists) ─────────────

print("\n" + "=" * 60)
print("STEP 8: Human agreement evaluation")
print("=" * 60)

if not os.path.exists(ANNOTATED_LONG):
    print(f"\nAnnotated file not found: {ANNOTATED_LONG}")
    print("\nPlease:")
    print("1. Open OUT_LONG")
    print("2. Fill human1 column with 0/1")
    print(f"3. Save as: {ANNOTATED_LONG}")

else:
    print(f"\nLoading annotated file: {ANNOTATED_LONG}")
    annotated_df = pd.read_csv(ANNOTATED_LONG)

    human_positive = annotated_df[annotated_df["human1"] == 1]
    human_type_sets = (
        human_positive
        .groupby(["reference", "hypothesis"])["error_class"]
        .apply(lambda x: frozenset(x.tolist()))
        .tolist()
    )

    total_records       = annotated_df[["reference", "hypothesis"]].drop_duplicates().shape[0]
    records_with_type   = len(human_type_sets)

    print(f"\n[human1] Critical records with ≥1 type: {records_with_type}")
    print(f"  Multi-label (>1 type): {sum(1 for t in human_type_sets if len(t) > 1)}")

    class_counts = Counter(cls for types in human_type_sets for cls in types)
    for cls, cnt in class_counts.most_common():
        print(f"  {cls:15s}: {cnt:5d} ({100 * cnt / records_with_type:.1f}%)")

    human1_wide = (
        annotated_df
        .pivot_table(index=["reference", "hypothesis"], columns="error_class", values="human1")
        .reset_index()
    )
    human1_wide.columns = [
        f"human1_{c}" if c not in ["reference", "hypothesis"] else c
        for c in human1_wide.columns
    ]

    eval_df = critical_df.merge(human1_wide, on=["reference", "hypothesis"], how="left", suffixes=("", "_new"))

    for cls in VALID_CLASSES:
        old_col = f"human1_{cls}"
        new_col = f"{old_col}_new"
        if new_col in eval_df.columns:
            eval_df.drop(columns=[old_col], inplace=True)
            eval_df.rename(columns={new_col: old_col}, inplace=True)

    print("\nPer-class agreement with human annotator:")
    print("=" * 60)

    for cls in VALID_CLASSES:
        print(f"\n[{cls}]")
        human_col = f"human1_{cls}"
        for judge_name in judge_names:
            judge_col = f"judge_{judge_name}_{cls}"
            subset    = eval_df[[judge_col, human_col]].dropna()
            if len(subset) < 2:
                print(f"  {judge_name:10s}: insufficient data")
                continue

            y_true = subset[human_col].astype(int)
            y_pred = subset[judge_col].astype(int)

            try:
                kappa = cohen_kappa_score(y_true, y_pred)
            except Exception:
                kappa = np.nan

            print(
                f"  {judge_name:10s} | "
                f"κ={kappa:6.3f} | "
                f"Acc={accuracy_score(y_true, y_pred):6.3f} | "
                f"P={precision_score(y_true, y_pred, zero_division=0):6.3f} | "
                f"R={recall_score(y_true, y_pred, zero_division=0):6.3f} | "
                f"F1={f1_score(y_true, y_pred, zero_division=0):6.3f} | "
                f"N={len(subset)}"
            )

print("\nDone.")
