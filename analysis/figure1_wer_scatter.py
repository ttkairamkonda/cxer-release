import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.join(_REPO, "annotations", "llama_70b")
DATASET  = "atcosim_test"

FILES = {
    "Whisper Medium (Baseline)":
        "whisper-medium_pretrained_" + DATASET + "_predictions.json",
    "Whisper Medium (Fine-tuned)":
        "whisper-medium-combined_" + DATASET + "_predictions.json",
}

COLOR = {"Critical_Errors": "#d62728", "Equivalent": "#2ca02c"}
rng   = np.random.default_rng(42)
N_EACH = 200

ANNOTATION_GREEN = {
    "title":  "High WER  →  Contextually Safe",
    "ref":    "alitalia 487 descend to flight level 270",
    "hyp":    "alitalia 487 descent fl 270",
    "xy":     (0.57, -0.07),
    "xytext": (0.20, 0.70),
}

ANNOTATION_RED = {
    "title":  "Low WER  →  Contextually Unsafe",
    "ref":    "jet aviation 701 is identified fly heading 250",
    "hyp":    "jet division 701 is identified fly heading 250",
    "xy":     (0.11, 0.01),
    "xytext": (0.20, 0.70),
}


def load_records(filename):
    with open(os.path.join(BASE_DIR, filename)) as f:
        return json.load(f)["records"]


def get_valid(records):
    return [
        r for r in records
        if r["contextual_status"] in ("Critical_Errors", "Equivalent")
        and r["WER"] <= 0.8
        and r["WER"] > 1e-8
    ]


baseline_file    = list(FILES.values())[0]
baseline_records = get_valid(load_records(baseline_file))
baseline_wers    = np.array([r["WER"] for r in baseline_records])
THRESHOLD        = np.percentile(baseline_wers, 25)
print(f"Fixed threshold (25th pct): {THRESHOLD:.3f}")

os.makedirs(os.path.join(_REPO, "eval_plots"), exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

for ax_idx, (ax, (title, filename)) in enumerate(zip(axes, FILES.items())):

    records    = get_valid(load_records(filename))
    critical   = [r for r in records if r["contextual_status"] == "Critical_Errors"]
    equivalent = [r for r in records if r["contextual_status"] == "Equivalent"]

    crit_wers = np.array([r["WER"] for r in critical])
    pct_crit_in_threshold = (
        100 * np.sum(crit_wers < THRESHOLD) / len(crit_wers)
        if len(crit_wers) else 0
    )

    n_crit = min(N_EACH, len(critical))
    n_eq   = min(N_EACH, len(equivalent))
    critical_plot   = list(rng.choice(critical,   size=n_crit, replace=False))
    equivalent_plot = list(rng.choice(equivalent, size=n_eq,   replace=False))

    plot_records = critical_plot + equivalent_plot
    wers         = np.array([r["WER"] for r in plot_records])
    statuses     = [r["contextual_status"] for r in plot_records]
    jitter       = rng.uniform(-0.4, 0.4, size=len(wers))

    ax.axvspan(0, THRESHOLD, color="#efefef", alpha=0.8, zorder=0)
    ax.axvline(THRESHOLD, color="#888888", linewidth=1.2, linestyle="--", zorder=2)
    ax.text(
        THRESHOLD / 2, 0.96,
        f"Lower 25%\n(WER ≤ {THRESHOLD:.2f})",
        transform=ax.get_xaxis_transform(),
        fontsize=7.5, color="#999999", va="top", ha="center", style="italic"
    )

    for status in ["Equivalent", "Critical_Errors"]:
        mask = np.array([s == status for s in statuses])
        ax.scatter(
            wers[mask], jitter[mask],
            c=COLOR[status], marker="o",
            alpha=0.4 if status == "Equivalent" else 0.7,
            s=14  if status == "Equivalent" else 20,
            linewidths=0
        )

    label_char = "(a)" if ax_idx == 0 else "(b)"
    ax.text(
        -0.01, 1.02,
        label_char,
        transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="bottom", ha="left"
    )

    if ax_idx == 0:
        ann = ANNOTATION_GREEN
        label = f"{ann['title']}\nRef:  {ann['ref']}\nHyp:  {ann['hyp']}"
        ax.annotate(
            label,
            xy=ann["xy"],
            xytext=ann["xytext"],
            fontsize=7,
            color="#1b5e20",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#2ca02c", lw=1.0),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.1),
        )

    if ax_idx == 1:
        ann = ANNOTATION_RED
        label = f"{ann['title']}\nRef:  {ann['ref']}\nHyp:  {ann['hyp']}"
        ax.annotate(
            label,
            xy=ann["xy"],
            xytext=ann["xytext"],
            fontsize=7,
            color="#8b0000",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d62728", lw=1.0),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1),
        )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=3)
    ax.set_xlim(-0.01, 0.82)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines[["top", "right", "left"]].set_visible(False)

axes[1].set_xlabel("Word Error Rate (WER)", fontsize=10)

fig.legend(
    handles=[
        mpatches.Patch(color=COLOR["Critical_Errors"], label="Critical Error"),
        mpatches.Patch(color=COLOR["Equivalent"],      label="Equivalent"),
    ],
    loc="upper center", ncol=2, frameon=False,
    bbox_to_anchor=(0.5, 1.02),
    fontsize=9,
)

plt.tight_layout()

out_path = os.path.join(_REPO, "eval_plots", f"wer_whisper_only_{DATASET}_vertical.png")
plt.savefig(out_path, dpi=600, bbox_inches="tight")
print(f"Saved: {out_path}")
