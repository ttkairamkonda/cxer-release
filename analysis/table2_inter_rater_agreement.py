import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from statsmodels.stats.inter_rater import fleiss_kappa

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LLM_FILE = os.path.join(_REPO, "eval_cache", "annotation_sheet_shared.xlsx")

POP_COUNTS = {
    "agreed_critical":   9988,
    "agreed_equivalent": 4729,
    "disagreed":         5822,
}
TOTAL_POP = sum(POP_COUNTS.values())

rng = np.random.default_rng(42)

df = pd.read_excel(LLM_FILE)
print(f"Loaded: {len(df)} rows")


def to_bool(x):
    return str(x).strip().lower() in {"true", "1", "1.0", "yes"}


mask = df["human_consensus_CCER"].notna()
df   = df[mask].reset_index(drop=True)
print(f"After dropping missing gold labels: {len(df)} rows")

j1            = df["judge1_llama_CCER"].map(to_bool)
j2            = df["judge2_qwen_CCER"].map(to_bool)
j3            = df["judge3_deepseek_CCER"].map(to_bool)
h1            = df["human1_CCER"].map(to_bool)
h2            = df["human2_CCER"].map(to_bool)
gold          = df["human_consensus_CCER"].map(to_bool)
llm_consensus = df["llm_consensus_CCER"].map(to_bool)

sample_counts = df["sample_type"].value_counts()
weights = df["sample_type"].map(
    lambda x: (POP_COUNTS[x] / TOTAL_POP) / (sample_counts[x] / len(df))
).values

print(f"\nWeight sanity check — unique weights: {np.unique(weights.round(4))}")
print(f"Weight sum: {weights.sum():.2f}  (expected ≈ {len(df)})")


def weighted_kappa(y1, y2, w):
    """Population-reweighted Cohen's kappa for binary labels."""
    y1 = np.array(y1, dtype=int)
    y2 = np.array(y2, dtype=int)
    w  = np.array(w,  dtype=float)

    cm = np.zeros((2, 2))
    for i in range(len(y1)):
        cm[y1[i], y2[i]] += w[i]

    n  = w.sum()
    Po = np.diag(cm).sum() / n
    Pe = ((cm.sum(axis=1) / n) @ (cm.sum(axis=0) / n))
    return (Po - Pe) / (1 - Pe)


def interpret(k):
    if k < 0:   return "Less than chance"
    if k < 0.2: return "Slight"
    if k < 0.4: return "Fair"
    if k < 0.6: return "Moderate"
    if k < 0.8: return "Substantial"
    return "Almost perfect"


def compute_metrics(name, y_true, y_pred):
    y_true    = np.array(y_true, dtype=int)
    y_pred    = np.array(y_pred, dtype=int)
    agreement = (y_true == y_pred).mean() * 100

    print(f"\n{name}")
    print("-" * len(name))
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Agreement: {agreement:.2f}%")


def bootstrap_kappa(y1, y2, n_bootstrap=1000):
    y1, y2 = np.array(y1), np.array(y2)
    kappas = []
    n      = len(y1)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        kappas.append(cohen_kappa_score(y1[idx], y2[idx]))
    return np.percentile(kappas, [2.5, 97.5])


# ── Weighted kappa ────────────────────────────────────────────────────────────

print("\n=== WEIGHTED KAPPA (population-reweighted) ===")
pairs_wk = [
    ("Human1 – Llama",    h1, j1),
    ("Human1 – Qwen",     h1, j2),
    ("Human1 – DeepSeek", h1, j3),
    ("Llama  – Qwen",     j1, j2),
    ("Llama  – DeepSeek", j1, j3),
    ("Qwen   – DeepSeek", j2, j3),
    ("Human1 – Human2",   h1, h2),
]
for label, a, b in pairs_wk:
    k = weighted_kappa(a, b, weights)
    print(f"  {label:<22}  κ = {k:.4f}  ({interpret(k)})")

# ── Cohen's kappa (unweighted) ────────────────────────────────────────────────

kappa_j1_j2 = cohen_kappa_score(j1, j2)
kappa_j1_j3 = cohen_kappa_score(j1, j3)
kappa_j2_j3 = cohen_kappa_score(j2, j3)
kappa_h1_j1 = cohen_kappa_score(h1, j1)
kappa_h1_j2 = cohen_kappa_score(h1, j2)
kappa_h1_j3 = cohen_kappa_score(h1, j3)
kappa_h2_j1 = cohen_kappa_score(h2, j1)
kappa_h2_j2 = cohen_kappa_score(h2, j2)
kappa_h2_j3 = cohen_kappa_score(h2, j3)
kappa_h1_h2 = cohen_kappa_score(h1, h2)

print("\n=== COHEN'S KAPPA (unweighted) ===")
print("\n  LLM vs LLM")
print(f"    Llama  – Qwen:     {kappa_j1_j2:.4f}  ({interpret(kappa_j1_j2)})")
print(f"    Llama  – DeepSeek: {kappa_j1_j3:.4f}  ({interpret(kappa_j1_j3)})")
print(f"    Qwen   – DeepSeek: {kappa_j2_j3:.4f}  ({interpret(kappa_j2_j3)})")

print("\n  Human1 vs LLM")
print(f"    Human1 – Llama:    {kappa_h1_j1:.4f}  ({interpret(kappa_h1_j1)})")
print(f"    Human1 – Qwen:     {kappa_h1_j2:.4f}  ({interpret(kappa_h1_j2)})")
print(f"    Human1 – DeepSeek: {kappa_h1_j3:.4f}  ({interpret(kappa_h1_j3)})")

print("\n  Human2 vs LLM")
print(f"    Human2 – Llama:    {kappa_h2_j1:.4f}  ({interpret(kappa_h2_j1)})")
print(f"    Human2 – Qwen:     {kappa_h2_j2:.4f}  ({interpret(kappa_h2_j2)})")
print(f"    Human2 – DeepSeek: {kappa_h2_j3:.4f}  ({interpret(kappa_h2_j3)})")

print("\n  Human vs Human")
print(f"    Human1 – Human2:   {kappa_h1_h2:.4f}  ({interpret(kappa_h1_h2)})")

# ── Bootstrap 95% CIs ────────────────────────────────────────────────────────

print("\n=== 95% CONFIDENCE INTERVALS ===")
ci_j1_j2 = bootstrap_kappa(j1, j2)
ci_j1_j3 = bootstrap_kappa(j1, j3)
ci_j2_j3 = bootstrap_kappa(j2, j3)
print(f"Llama-Qwen:     [{ci_j1_j2[0]:.4f}, {ci_j1_j2[1]:.4f}]")
print(f"Llama-DeepSeek: [{ci_j1_j3[0]:.4f}, {ci_j1_j3[1]:.4f}]")
print(f"Qwen-DeepSeek:  [{ci_j2_j3[0]:.4f}, {ci_j2_j3[1]:.4f}]")

# ── Fleiss kappa (3 LLMs) ────────────────────────────────────────────────────

def weighted_fleiss_kappa(rating_matrix, w):
    """Population-reweighted Fleiss kappa."""
    w = np.array(w, dtype=float)
    N, k = rating_matrix.shape
    n    = rating_matrix[0].sum()

    W   = w.sum()
    p_j = (rating_matrix * w[:, None]).sum(axis=0) / (W * n)

    P_i    = ((rating_matrix ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar  = (w * P_i).sum() / W
    P_e    = (p_j ** 2).sum()

    return (P_bar - P_e) / (1 - P_e)


def build_fleiss_llm(dataframe):
    N      = len(dataframe)
    matrix = np.zeros((N, 2), dtype=float)
    for i, (_, row) in enumerate(dataframe.iterrows()):
        judges = [
            to_bool(row["judge1_llama_CCER"]),
            to_bool(row["judge2_qwen_CCER"]),
            to_bool(row["judge3_deepseek_CCER"]),
        ]
        matrix[i, 0] = sum(not j for j in judges)
        matrix[i, 1] = sum(j for j in judges)
    return matrix


llm_matrix             = build_fleiss_llm(df)
fleiss_llm_unweighted  = fleiss_kappa(llm_matrix, method="fleiss")
fleiss_llm_weighted    = weighted_fleiss_kappa(llm_matrix, weights)

print(f"\n=== FLEISS κ (3 LLMs) ===")
print(f"  Unweighted (stratified sample):      {fleiss_llm_unweighted:.4f}  ({interpret(fleiss_llm_unweighted)})")
print(f"  Weighted   (population-reweighted):  {fleiss_llm_weighted:.4f}  ({interpret(fleiss_llm_weighted)})")

# ── vs Human Consensus ───────────────────────────────────────────────────────

print("\n=== vs HUMAN CONSENSUS (GOLD) ===")
compute_metrics("Llama        vs Gold", gold, j1)
compute_metrics("Qwen         vs Gold", gold, j2)
compute_metrics("DeepSeek     vs Gold", gold, j3)
compute_metrics("LLM Consensus vs Gold", gold, llm_consensus)

kappa_consensus = cohen_kappa_score(gold, llm_consensus)
print(f"Cohen's κ (LLM Consensus vs Human Consensus): {kappa_consensus:.4f}")
print(f"Weighted κ (LLM Consensus vs Human Consensus): {weighted_kappa(gold, llm_consensus, weights):.4f}")

# ── Kappa heatmap (Appendix C) ───────────────────────────────────────────────

judge_names  = ["Llama", "Qwen", "DeepSeek", "Human1", "Human2"]
kappa_matrix = np.array([
    [1.0,         kappa_j1_j2,  kappa_j1_j3,  kappa_h1_j1,  kappa_h2_j1],
    [kappa_j1_j2, 1.0,          kappa_j2_j3,  kappa_h1_j2,  kappa_h2_j2],
    [kappa_j1_j3, kappa_j2_j3,  1.0,          kappa_h1_j3,  kappa_h2_j3],
    [kappa_h1_j1, kappa_h1_j2,  kappa_h1_j3,  1.0,          kappa_h1_h2],
    [kappa_h2_j1, kappa_h2_j2,  kappa_h2_j3,  kappa_h1_h2,  1.0        ],
])

assert np.allclose(kappa_matrix, kappa_matrix.T), "Matrix is not symmetric!"

df_kappa   = pd.DataFrame(kappa_matrix, index=judge_names, columns=judge_names)
mask_upper = np.triu(np.ones_like(df_kappa, dtype=bool))

os.makedirs(os.path.join(_REPO, "eval_plots"), exist_ok=True)

plt.figure(figsize=(8, 7))
sns.heatmap(
    df_kappa,
    mask=mask_upper,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Cohen's κ"},
    vmin=0,
    vmax=1,
)
plt.title(
    f"Inter-Rater Agreement  (Fleiss' κ = {fleiss_llm_unweighted:.2f})",
    fontsize=13,
    fontweight="bold",
)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()

out_path = os.path.join(_REPO, "eval_plots", "kappa_heatmap.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nSaved heatmap → {out_path}")
