import os
import pandas as pd
import json
import uuid

# ============================================================
# INPUT FILES
# ============================================================

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LLM_FILE   = os.path.join(_REPO, "eval_cache", "annotation_sheet_shared.csv")
HUMAN_FILE = os.path.join(_REPO, "eval_cache", "annotation_sheet_shared.xlsx")

# ============================================================
# LOAD
# ============================================================

df_llm = pd.read_csv(LLM_FILE)
df_llm = df_llm.drop(columns=["human2_CCER"])
df_human = pd.read_excel(HUMAN_FILE)

# ============================================================
# NORMALIZE KEYS
# ============================================================

def make_key(df):
    return (
        df["reference"].astype(str).str.strip().str.lower()
        + "|||"
        + df["hypothesis"].astype(str).str.strip().str.lower()
    )

df_llm["key"] = make_key(df_llm)
df_human["key"] = make_key(df_human)

# ============================================================
# MERGE HUMAN LABELS
# ============================================================

merged = df_llm.merge(
    df_human[["key", "human2_CCER"]],
    on="key",
    how="inner"
)

print(f"Merged samples: {len(merged)}")

# ============================================================
# SAFE BOOLEAN CONVERSION
# ============================================================

def to_bool(x):
    return str(x).strip().lower() in [
        "true",
        "1",
        "yes"
    ]

# ============================================================
# CONVERT LABELS
# ============================================================

merged["judge1_llama_CCER"] = merged["judge1_llama_CCER"].map(to_bool)
merged["judge2_qwen_CCER"] = merged["judge2_qwen_CCER"].map(to_bool)
merged["judge3_deepseek_CCER"] = merged["judge3_deepseek_CCER"].map(to_bool)

merged["human_CCER"] = merged["human2_CCER"].map(to_bool)

# ============================================================
# OPTIONAL: SAMPLE BALANCED SUBSET
# ============================================================

N_PER_CLASS = 100

balanced_df = pd.concat([
    merged[merged["sample_type"] == "agreed_critical"]
    .sample(min(N_PER_CLASS,
                len(merged[merged["sample_type"] == "agreed_critical"])),
            random_state=42),

    merged[merged["sample_type"] == "agreed_equivalent"]
    .sample(min(N_PER_CLASS,
                len(merged[merged["sample_type"] == "agreed_equivalent"])),
            random_state=42),

    merged[merged["sample_type"] == "disagreed"]
    .sample(min(N_PER_CLASS,
                len(merged[merged["sample_type"] == "disagreed"])),
            random_state=42)
])

balanced_df = balanced_df.reset_index(drop=True)

print(f"Balanced subset size: {len(balanced_df)}")

# ============================================================
# PREPARE JSON RECORDS
# ============================================================

records = []

for idx, row in balanced_df.iterrows():

    record = {

        # --------------------------------------------
        # UNIQUE ID
        # --------------------------------------------

        "sample_id": str(uuid.uuid4()),

        # --------------------------------------------
        # TEXT
        # --------------------------------------------

        "reference": row["reference"],
        "hypothesis": row["hypothesis"],

        # --------------------------------------------
        # DATASET METADATA
        # --------------------------------------------

        "sample_type": row["sample_type"],

        # --------------------------------------------
        # BASELINE LLM JUDGMENTS
        # --------------------------------------------

        "baseline_labels": {

            "llama": bool(row["judge1_llama_CCER"]),
            "qwen": bool(row["judge2_qwen_CCER"]),
            "deepseek": bool(row["judge3_deepseek_CCER"])
        },

        # --------------------------------------------
        # HUMAN LABEL
        # --------------------------------------------

        "human_label": bool(row["human_CCER"]),

        # --------------------------------------------
        # PLACEHOLDER FOR FUTURE PERTURBATION RUNS
        # --------------------------------------------

        "perturbation_results": {

            # Example future structure:
            #
            # "compressed_prompt": {
            #     "llama": True,
            #     "qwen": False,
            #     "deepseek": True
            # }

        }
    }

    records.append(record)

# ============================================================
# SAVE JSON
# ============================================================

OUT_JSON = os.path.join(_REPO, "eval_cache", "prompt_robustness_benchmark.json")

with open(OUT_JSON, "w") as f:
    json.dump(
        records,
        f,
        indent=2
    )

print(f"\nSaved benchmark JSON:")
print(OUT_JSON)

# ============================================================
# QUICK SUMMARY
# ============================================================

print("\nClass distribution:")

print(
    balanced_df["sample_type"]
    .value_counts()
)

print("\nExample record:\n")

print(
    json.dumps(
        records[0],
        indent=2
    )
)