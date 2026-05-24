"""
Minimal Prompt Robustness Analysis for Short Paper
===================================================
Generates ONLY the essential metrics:
1. Table: F1 scores across prompts (with mean±std)
2. Table: Inter-prompt agreement (Fleiss' Kappa)
3. Figure: Accuracy-Robustness scatter plot
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_ROOT     = os.path.join(_REPO, "prompt_robustness_outputs")
BENCHMARK_JSON  = os.path.join(_REPO, "eval_cache", "prompt_robustness_benchmark.json")
EXCEL_PATH      = os.path.join(_REPO, "eval_cache", "annotation_sheet_shared.xlsx")
ANALYSIS_OUTPUT = os.path.join(_REPO, "prompt_robustness_analysis")

MODELS  = ["llama", "qwen", "deepseek"]
PROMPTS = ["cot_v3", "compressed", "no_examples", "lenient"]


def load_ground_truth():
    df = pd.read_excel(EXCEL_PATH)
    with open(BENCHMARK_JSON) as f:
        benchmark = json.load(f)

    gt_lookup = {
        (row["reference"], row["hypothesis"]): row["human_consensus_CCER"]
        for _, row in df.iterrows()
    }

    return [gt_lookup.get((s["reference"], s["hypothesis"])) for s in benchmark]


def load_predictions(model, prompt):
    path = Path(OUTPUT_ROOT) / prompt / f"{model}.json"
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    preds = [None] * 300
    for rec in data["records"]:
        idx = rec["sample_index"]
        if idx < 300:
            preds[idx] = rec.get("is_critical")

    return preds


def fleiss_kappa(matrix):
    n_samples, n_categories = matrix.shape
    n_raters = matrix.sum(axis=1)[0]

    p_j   = matrix.sum(axis=0) / (n_samples * n_raters)
    P_i   = (matrix ** 2).sum(axis=1) - n_raters
    P_i   = P_i / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e   = (p_j ** 2).sum()

    return (P_bar - P_e) / (1 - P_e)


def main():
    os.makedirs(ANALYSIS_OUTPUT, exist_ok=True)

    print("=" * 60)
    print("PROMPT ROBUSTNESS ANALYSIS (MINIMAL)")
    print("=" * 60)

    gt = load_ground_truth()

    # ── Table: F1 scores ─────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("TABLE 1: ACCURACY UNDER PROMPT PERTURBATION")
    print("=" * 60)

    f1_table = []

    for model in MODELS:
        row      = {"model": model}
        f1_scores = []

        for prompt in PROMPTS:
            preds = load_predictions(model, prompt)
            if preds is None:
                continue

            valid_pairs = [(p, g) for p, g in zip(preds, gt) if g is not None and p is not None and not (isinstance(g, float) and np.isnan(g))]
            if not valid_pairs:
                continue

            p, g = zip(*valid_pairs)
            _, _, f1, _ = precision_recall_fscore_support(g, p, average="binary", zero_division=0)

            row[prompt] = f1
            f1_scores.append(f1)

        if f1_scores:
            row["mean"] = np.mean(f1_scores)
            row["std"]  = np.std(f1_scores)

        f1_table.append(row)

    f1_df = pd.DataFrame(f1_table)

    print("\n{:<12} {:<10} {:<12} {:<13} {:<10} {:<12}".format(
        "Model", "CoT_v3", "Compressed", "No_Examples", "Lenient", "Mean±Std"
    ))
    print("-" * 80)
    for _, row in f1_df.iterrows():
        print("{:<12} {:<10.3f} {:<12.3f} {:<13.3f} {:<10.3f} {:.3f}±{:.3f}".format(
            row["model"].capitalize(),
            row.get("cot_v3",      0),
            row.get("compressed",  0),
            row.get("no_examples", 0),
            row.get("lenient",     0),
            row.get("mean",        0),
            row.get("std",         0),
        ))

    latex_table = f1_df.to_latex(
        index=False,
        float_format="%.3f",
        columns=["model", "cot_v3", "compressed", "no_examples", "lenient", "mean", "std"],
    )
    with open(os.path.join(ANALYSIS_OUTPUT, "table1_f1_scores.tex"), "w") as f:
        f.write(latex_table)

    f1_df.to_csv(os.path.join(ANALYSIS_OUTPUT, "table1_f1_scores.csv"), index=False)
    print(f"\nSaved: table1_f1_scores.csv and .tex")

    # ── Table: Inter-prompt agreement ─────────────────────────────────────────

    print("\n" + "=" * 60)
    print("TABLE 2: INTER-PROMPT AGREEMENT")
    print("=" * 60)

    agreement_table = []

    for model in MODELS:
        all_preds = {}
        for prompt in PROMPTS:
            preds = load_predictions(model, prompt)
            if preds:
                all_preds[prompt] = preds

        if len(all_preds) < 2:
            continue

        pred_matrix   = np.array([all_preds[p] for p in PROMPTS if p in all_preds]).T
        complete_mask  = ~pd.isna(pred_matrix).any(axis=1)
        pred_complete  = pred_matrix[complete_mask].astype(int)

        if len(pred_complete) == 0:
            continue

        n_prompts  = pred_complete.shape[1]
        agreements = [
            (pred_complete[:, i] == pred_complete[:, j]).mean()
            for i in range(n_prompts)
            for j in range(i + 1, n_prompts)
        ]
        pairwise_mean = np.mean(agreements)

        kappa_matrix = np.zeros((len(pred_complete), 2))
        for i, row in enumerate(pred_complete):
            kappa_matrix[i, 0] = (row == 0).sum()
            kappa_matrix[i, 1] = (row == 1).sum()

        kappa = fleiss_kappa(kappa_matrix)

        agreement_table.append({
            "model":              model,
            "fleiss_kappa":       kappa,
            "pairwise_agreement": pairwise_mean,
            "n_samples":          len(pred_complete),
        })

    agreement_df = pd.DataFrame(agreement_table)

    print("\n{:<12} {:<15} {:<20}".format("Model", "Fleiss' κ", "Pairwise Agreement"))
    print("-" * 50)
    for _, row in agreement_df.iterrows():
        print("{:<12} {:<15.3f} {:<20.3f}".format(
            row["model"].capitalize(),
            row["fleiss_kappa"],
            row["pairwise_agreement"],
        ))

    agreement_df.to_csv(os.path.join(ANALYSIS_OUTPUT, "table2_agreement.csv"), index=False)

    latex_agreement = agreement_df[["model", "fleiss_kappa", "pairwise_agreement"]].to_latex(
        index=False, float_format="%.3f"
    )
    with open(os.path.join(ANALYSIS_OUTPUT, "table2_agreement.tex"), "w") as f:
        f.write(latex_agreement)

    print(f"\nSaved: table2_agreement.csv and .tex")

    # ── Figure: Accuracy-Robustness scatter ───────────────────────────────────

    os.makedirs(os.path.join(_REPO, "eval_plots"), exist_ok=True)

    plot_data = f1_df[["model", "mean"]].merge(
        agreement_df[["model", "fleiss_kappa"]], on="model"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = {"llama": "#1f77b4", "qwen": "#ff7f0e", "deepseek": "#2ca02c"}

    for _, row in plot_data.iterrows():
        ax.scatter(
            row["mean"],
            row["fleiss_kappa"],
            s=200,
            c=colors[row["model"]],
            label=row["model"].capitalize(),
            alpha=0.7,
            edgecolors="black",
            linewidth=1.5,
        )

    ax.axhline(y=0.65, color="gray",  linestyle="--", alpha=0.5, label="Human-Human κ=0.65")
    ax.axhline(y=0.60, color="red",   linestyle=":",  alpha=0.3)
    ax.axvline(x=0.85, color="green", linestyle=":",  alpha=0.3)

    ax.set_xlabel("Mean F1 Score (Accuracy)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fleiss' Kappa (Robustness)", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy-Robustness Tradeoff\nAcross Prompt Variants", fontsize=14, fontweight="bold")

    ax.text(0.83, 0.78, "Ideal\n(Accurate & Robust)",
            fontsize=9, ha="center",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3))

    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0.72, 0.88)
    ax.set_ylim(0.30, 0.85)

    plt.tight_layout()

    out_path = os.path.join(_REPO, "eval_plots", "figure_tradeoff.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(_REPO, "eval_plots", "figure_tradeoff.pdf"), bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    best_f1    = f1_df.loc[f1_df["mean"].idxmax()]
    most_robust = agreement_df.loc[agreement_df["fleiss_kappa"].idxmax()]

    print(f"\nHighest Mean F1: {best_f1['model'].capitalize()} ({best_f1['mean']:.3f})")
    print(f"Most Robust (κ): {most_robust['model'].capitalize()} ({most_robust['fleiss_kappa']:.3f})")
    print(f"\nReference: Human-Human κ = 0.65")


if __name__ == "__main__":
    main()
