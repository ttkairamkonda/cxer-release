import os
import json
import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.join(_REPO, "annotations", "llama_70b")

DATASETS = ["atco2_ood", "uwb_atcc_test", "atcosim_test"]
MODELS   = ["whisper-medium", "parakeet-tdt-0.6b-v3"]
VARIANTS = ["pretrained", "combined"]

MODEL_LABEL = {
    "whisper-medium":        "Whisper",
    "parakeet-tdt-0.6b-v3": "Parakeet",
}

DATASET_LABEL = {
    "atco2_ood":     "ATCO2 (OOD)",
    "uwb_atcc_test": "UWB-ATCC (ID)",
    "atcosim_test":  "ATCOSim (ID)",
}


def load(file):
    path = os.path.join(BASE_DIR, file)
    with open(path) as f:
        return json.load(f)["records"]


def get_valid(records):
    return [
        r for r in records
        if r["contextual_status"] in ("Critical_Errors", "Equivalent")
        and r.get("WER") is not None
    ]


all_wers = []
for ds in DATASETS:
    for model in MODELS:
        file = f"{model}_pretrained_{ds}_predictions.json"
        path = os.path.join(BASE_DIR, file)
        if not os.path.exists(path):
            continue
        recs = get_valid(load(file))
        all_wers.extend([r["WER"] for r in recs])

THRESHOLD = np.percentile(all_wers, 25)
print(f"Global WER threshold (25th percentile): {THRESHOLD:.4f}")

rows = []
for ds in DATASETS:
    for model in MODELS:
        for var in VARIANTS:
            file = f"{model}_{var}_{ds}_predictions.json"
            path = os.path.join(BASE_DIR, file)
            if not os.path.exists(path):
                continue

            records  = get_valid(load(file))
            critical = [r for r in records if r["contextual_status"] == "Critical_Errors"]

            if len(critical) == 0:
                pct_low = np.nan
            else:
                low     = [r for r in critical if r["WER"] < THRESHOLD]
                pct_low = 100 * len(low) / len(critical)

            rows.append({
                "Dataset":                       DATASET_LABEL[ds],
                "Model":                         MODEL_LABEL[model],
                "Variant":                       var,
                "Critical Samples":              len(critical),
                "% Critical in Low-WER Region":  round(pct_low, 2),
            })

df = pd.DataFrame(rows)
print("\n")
print(df.to_string(index=False))
