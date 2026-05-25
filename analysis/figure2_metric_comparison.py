"""
ccer_improvement_gap.py
=======================
Original grouped bar chart (pretrained vs fine-tuned) with % improvement
annotations bracketed between each model pair. Clean, uncluttered.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(_REPO, "eval_cache", "all_results.csv"))
df = df[df["judge"] == "qwen"]

DATASETS = ["atco2_ood", "uwb_atcc_test", "atcosim_test"]
DATASET_LABELS = {
    "atco2_ood":     "ATCO2",
    "uwb_atcc_test": "UWB-ATCC",
    "atcosim_test":  "ATCOSim",
}

PAIRS = [
    ("whisper-medium",        "pretrained", "combined"),
    ("parakeet-tdt-0.6b-v3", "pretrained", "combined"),
]

MODEL_ORDER = [
    ("whisper-medium",        "pretrained"),
    ("whisper-medium",        "combined"),
    ("parakeet-tdt-0.6b-v3", "pretrained"),
    ("parakeet-tdt-0.6b-v3", "combined"),
]

MODEL_LABELS = {
    ("whisper-medium",        "pretrained"): "Whisper\nPretrained",
    ("whisper-medium",        "combined"):   "Whisper\nFine-tuned",
    ("parakeet-tdt-0.6b-v3", "pretrained"): "Parakeet\nPretrained",
    ("parakeet-tdt-0.6b-v3", "combined"):   "Parakeet\nFine-tuned",
}

COLORS = {
    ("whisper-medium",        "pretrained"): "#4C72B0",
    ("whisper-medium",        "combined"):   "#76A0D4",
    ("parakeet-tdt-0.6b-v3", "pretrained"): "#2E8B57",
    ("parakeet-tdt-0.6b-v3", "combined"):   "#7BC8A4",
}

METRICS = {
    "WER":         "Word Error Rate",
    "WeightedWER": "Weighted WER (entity w=3)",
    "BERTDist":    "BERT Distance",
    "SemDist":     "Semantic Distance",
    "CCER":        "Contextual Critical Error Rate (CxER)",
}


def annotate_improvement(ax, x_pre, x_ft, y_pre, y_ft, pct, is_ccer=False):
    x_mid  = (x_pre + x_ft) / 2
    y_top  = max(y_pre, y_ft)
    cap    = y_top * 0.07
    y_line = y_top + cap * 0.6

    color = "#B22222" if is_ccer else "#333333"
    lw    = 1.1

    ax.plot([x_pre, x_pre], [y_pre, y_line], color=color, lw=lw, clip_on=False)
    ax.plot([x_ft,  x_ft],  [y_ft,  y_line], color=color, lw=lw, clip_on=False)
    ax.plot([x_pre, x_ft],  [y_line, y_line], color=color, lw=lw, clip_on=False)

    label = f"{abs(pct):.1f}%"

    ax.text(
        x_mid,
        y_line + cap * 0.3,
        label,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        clip_on=False,
    )


os.makedirs(os.path.join(_REPO, "eval_plots"), exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
fig.patch.set_facecolor("#FAFAFA")
axes = axes.flatten()
axes[-1].set_visible(False)   # 6th slot unused

bar_width = 0.18
x         = np.arange(len(DATASETS))
gap       = 0.04

offsets = {
    ("whisper-medium",        "pretrained"): -1.5 * bar_width - gap,
    ("whisper-medium",        "combined"):   -0.5 * bar_width - gap,
    ("parakeet-tdt-0.6b-v3", "pretrained"):  0.5 * bar_width + gap,
    ("parakeet-tdt-0.6b-v3", "combined"):    1.5 * bar_width + gap,
}

for ax, (metric, ylabel) in zip(axes, METRICS.items()):

    is_ccer = (metric == "CCER")
    is_wwer = (metric == "WeightedWER")
    bar_map = {}

    for model_key in MODEL_ORDER:
        model, variant = model_key
        vals = []
        for ds in DATASETS:
            row = df[
                (df["model"] == model) &
                (df["variant"] == variant) &
                (df["dataset"] == ds)
            ]
            vals.append(row[metric].values[0])

        xs   = x + offsets[model_key]
        bars = ax.bar(
            xs, vals,
            width=bar_width,
            label=MODEL_LABELS[model_key],
            color=COLORS[model_key],
            edgecolor="black",
            linewidth=0.5,
        )

        for i, (bar, val) in enumerate(zip(bars, vals)):
            bar_map[(model_key, i)] = (bar.get_x() + bar.get_width() / 2, val)

    for model, pre_var, ft_var in PAIRS:
        pre_key = (model, pre_var)
        ft_key  = (model, ft_var)
        for ds_idx, ds in enumerate(DATASETS):
            x_pre, h_pre = bar_map[(pre_key, ds_idx)]
            x_ft,  h_ft  = bar_map[(ft_key,  ds_idx)]

            pre_val = df[
                (df["model"] == model) & (df["variant"] == pre_var) & (df["dataset"] == ds)
            ][metric].values[0]
            ft_val = df[
                (df["model"] == model) & (df["variant"] == ft_var) & (df["dataset"] == ds)
            ][metric].values[0]

            pct = (pre_val - ft_val) / pre_val * 100.0

            annotate_improvement(ax, x_pre, x_ft, h_pre, h_ft, pct, is_ccer=is_ccer or is_wwer)

    ax.axvspan(-0.5, 0.5, color="#F6D6D6", alpha=0.35, zorder=0)
    ax.axvspan(0.5,  2.5, color="#DCEFD9", alpha=0.35, zorder=0)

    ax.text(0,   0.995, "OOD",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color="#B22222", fontweight="bold")
    ax.text(1.5, 0.995, "In-Distribution",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color="#2E7D32", fontweight="bold")

    title_color = "#B22222" if (is_ccer or is_wwer) else "#1A1A2E"
    ax.set_title(ylabel, fontsize=10, fontweight="bold", color=title_color, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#F7F9FC")

    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.25)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=4,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.04),
    framealpha=0.9,
    edgecolor="#CCC",
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

out_path = os.path.join(_REPO, "eval_plots", "ccer_improvement_gap.png")
plt.savefig(
    out_path,
    dpi=600,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
print(f"Saved: {out_path}")
