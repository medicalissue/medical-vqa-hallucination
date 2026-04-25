"""Generate per-dataset plots tailored for the handoff folder."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
df = pd.read_csv(ROOT / "results" / "full_v2" / "summary_long.csv")
HANDOFF = ROOT / "handoff"

DATASETS = ["vqa_rad", "vqa_med_2019", "vqa_med_2021"]
DS_LABEL_KO = {"vqa_rad": "VQA-RAD", "vqa_med_2019": "VQA-Med 2019", "vqa_med_2021": "VQA-Med 2021"}
DS_SHORT = {"vqa_rad": "vqarad", "vqa_med_2019": "med2019", "vqa_med_2021": "med2021"}
COLORS = {"biomed_clip": "#4080c0", "llava_med": "#d04040"}
MODELS = ["biomed_clip", "llava_med"]


def per_dataset_p3_metrics(d):
    """One panel — bar chart of P3 flip across 4 metrics for both models."""
    sub = df[(df["probe"] == "P3_flip") & (df["dataset"] == d)]
    metrics = ["naive", "yes_no", "jaccard", "embedding"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metrics)); w = 0.38
    for i, m in enumerate(MODELS):
        rows = sub[sub["model"] == m].set_index("metric").reindex(metrics)
        vals = rows["value"].fillna(0).values
        ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=COLORS[m])
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 0.5) * w, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05); ax.set_ylabel("flip rate")
    ax.legend(); ax.set_title(f"P3 — irrelevant prefix flip rate × metric ({DS_LABEL_KO[d]})")
    fig.tight_layout(); return fig


def baseline_yes_no(d):
    sub = df[(df["probe"] == "baseline") & (df["metric"] == "yes_no") & (df["dataset"] == d)]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, m in enumerate(MODELS):
        r = sub[sub["model"] == m]
        if r.empty: continue
        v = r["value"].iloc[0]
        n = int(r["n"].iloc[0])
        lo, hi = r["ci_lo"].iloc[0], r["ci_hi"].iloc[0]
        ax.bar([m], [v], color=COLORS[m],
               yerr=[[max(0, v - (lo if not pd.isna(lo) else v))],
                     [max(0, (hi if not pd.isna(hi) else v) - v)]], capsize=4)
        ax.text(i, v + 0.02, f"{v:.0%}\n(n_closed={n})", ha="center")
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy")
    ax.set_title(f"Baseline accuracy (closed yes/no only) — {DS_LABEL_KO[d]}")
    fig.tight_layout(); return fig


def baseline_lenient(d):
    sub = df[(df["probe"] == "baseline") & (df["metric"] == "lenient") & (df["dataset"] == d)]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, m in enumerate(MODELS):
        r = sub[sub["model"] == m]
        if r.empty: continue
        v = r["value"].iloc[0]
        n = int(r["n"].iloc[0])
        lo, hi = r["ci_lo"].iloc[0], r["ci_hi"].iloc[0]
        ax.bar([m], [v], color=COLORS[m],
               yerr=[[max(0, v - (lo if not pd.isna(lo) else v))],
                     [max(0, (hi if not pd.isna(hi) else v) - v)]], capsize=4)
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center")
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy")
    ax.set_title(f"Baseline accuracy (lenient match) — {DS_LABEL_KO[d]}")
    fig.tight_layout(); return fig


def p4_cross_one(d):
    sub = df[(df["probe"] == "P4_cross_change") & (df["dataset"] == d)]
    metrics = ["naive", "jaccard", "embedding"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(metrics)); w = 0.38
    for i, m in enumerate(MODELS):
        rows = sub[sub["model"] == m].set_index("metric").reindex(metrics)
        vals = rows["value"].fillna(0).values
        ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=COLORS[m])
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 0.5) * w, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05); ax.set_ylabel("cross-demographic change rate")
    ax.legend(); ax.set_title(f"P4 — Cross-demographic change × metric ({DS_LABEL_KO[d]})")
    fig.tight_layout(); return fig


def main():
    img_dir = HANDOFF / "03_데이터셋별_결과" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for d in DATASETS:
        s = DS_SHORT[d]
        per_dataset_p3_metrics(d).savefig(img_dir / f"P3_metrics_compare_{s}.png", dpi=140); plt.close()
        baseline_yes_no(d).savefig(img_dir / f"baseline_yes_no_{s}.png", dpi=140); plt.close()
        baseline_lenient(d).savefig(img_dir / f"baseline_lenient_{s}.png", dpi=140); plt.close()
        p4_cross_one(d).savefig(img_dir / f"P4_cross_{s}.png", dpi=140); plt.close()
        # P3 embedding
        sub = df[(df["probe"] == "P3_flip") & (df["metric"] == "embedding") & (df["dataset"] == d)]
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for i, m in enumerate(MODELS):
            r = sub[sub["model"] == m]
            if r.empty: continue
            v = r["value"].iloc[0]
            ax.bar([m], [v], color=COLORS[m])
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"P3 flip — embedding (semantic) — {DS_LABEL_KO[d]}")
        fig.tight_layout()
        fig.savefig(img_dir / f"P3_flip_embedding_{s}.png", dpi=140); plt.close()
    print(f"wrote per-dataset plots to {img_dir}")
    # also rename existing key plots into 03 with _vqarad / _med2019 / _med2021 suffixes
    # if needed, link from main full_v2 results
    print("done.")


if __name__ == "__main__":
    main()
