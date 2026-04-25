"""Generate plots from full_v2/summary_long.csv (multi-metric, with CIs)."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
df = pd.read_csv(ROOT / "results" / "full_v2" / "summary_long.csv")

PLOTS = ROOT / "results" / "full_v2" / "plots"
PLOTS.mkdir(exist_ok=True)
DATASETS = ["vqa_rad", "vqa_med_2019", "vqa_med_2021"]
MODELS = ["biomed_clip", "llava_med"]
COLORS = {"biomed_clip": "#4080c0", "llava_med": "#d04040"}


def _plot_grouped(probe: str, metric: str, title: str, fname: str):
    sub = df[(df["probe"] == probe) & (df["metric"] == metric)]
    if sub.empty: return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(DATASETS)); w = 0.38
    for i, m in enumerate(MODELS):
        rows = sub[sub["model"] == m].set_index("dataset").reindex(DATASETS)
        vals = rows["value"].fillna(0).values
        lo = rows["ci_lo"].fillna(np.nan).values
        hi = rows["ci_hi"].fillna(np.nan).values
        err_lo = np.clip(vals - np.where(np.isnan(lo), vals, lo), 0, None)
        err_hi = np.clip(np.where(np.isnan(hi), vals, hi) - vals, 0, None)
        ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=COLORS[m],
               yerr=[err_lo, err_hi], capsize=4)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 0.5) * w, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(DATASETS); ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate / accuracy"); ax.legend()
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=140); plt.close(fig)


def _plot_metric_grid(probe: str, title: str, fname: str, metrics: list):
    """A 2x2 grid showing the same probe under different metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, met in zip(axes.flat, metrics):
        sub = df[(df["probe"] == probe) & (df["metric"] == met)]
        x = np.arange(len(DATASETS)); w = 0.38
        for i, m in enumerate(MODELS):
            rows = sub[sub["model"] == m].set_index("dataset").reindex(DATASETS)
            vals = rows["value"].fillna(0).values
            ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=COLORS[m])
            for j, v in enumerate(vals):
                ax.text(x[j] + (i - 0.5) * w, v + 0.02, f"{v:.0%}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(DATASETS, fontsize=8); ax.set_ylim(0, 1.05)
        ax.set_title(f"metric: {met}")
    axes[0,0].set_ylabel("flip rate")
    axes[1,0].set_ylabel("flip rate")
    axes[0,1].legend()
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=140); plt.close(fig)


def _plot_p1_kinds(metric: str, fname: str):
    """One panel per blank kind (black/white/gray/noise) under given metric."""
    kinds = ("black", "white", "gray", "noise")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, kind in zip(axes, kinds):
        sub = df[(df["probe"] == f"P1_{kind}_flip") & (df["metric"] == metric)]
        x = np.arange(len(DATASETS)); w = 0.38
        for i, m in enumerate(MODELS):
            rows = sub[sub["model"] == m].set_index("dataset").reindex(DATASETS)
            vals = rows["value"].fillna(0).values
            ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=COLORS[m])
            for j, v in enumerate(vals):
                ax.text(x[j] + (i - 0.5) * w, v + 0.02, f"{v:.0%}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(DATASETS, fontsize=7); ax.set_ylim(0, 1.05)
        ax.set_title(kind)
    axes[0].set_ylabel("flip rate")
    axes[-1].legend()
    fig.suptitle(f"P1 — answer flip rate for blank-image variants ({metric} metric)")
    fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=140); plt.close(fig)


def _plot_demographic(metric: str, dataset: str, fname: str):
    """Per-demographic accuracy under given metric for a single dataset."""
    sub = df[(df["dataset"] == dataset) &
             (df["probe"].str.startswith("P4_demo_")) &
             (df["metric"] == metric)]
    if sub.empty: return
    sub = sub.copy()
    sub["demo"] = sub["probe"].str.replace("P4_demo_", "")
    demos = sorted(sub["demo"].unique())
    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(demos)); w = 0.38
    for i, m in enumerate(MODELS):
        rows = sub[sub["model"] == m].set_index("demo").reindex(demos)
        vals = rows["value"].fillna(0).values
        ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels(demos, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title(f"P4 — Per-demographic accuracy ({metric}) — {dataset}")
    fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=140); plt.close(fig)


def main():
    # Headline metrics with embedding-based flip
    _plot_grouped("baseline", "lenient",
                  "Baseline accuracy (lenient match) — original image+question",
                  "baseline_lenient.png")
    _plot_grouped("baseline", "yes_no",
                  "Baseline accuracy (closed yes/no only)", "baseline_yes_no.png")
    _plot_grouped("P2", "halluc",
                  "P2 — Confident hallucination on image-text mismatch (refusal == 1 - this)",
                  "P2_halluc.png")
    for met in ("naive", "yes_no", "jaccard", "embedding"):
        _plot_grouped("P3_flip", met,
                      f"P3 — Irrelevant prefix flip rate ({met} metric)",
                      f"P3_flip_{met}.png")
    _plot_metric_grid("P3_flip", "P3 — Prefix flip across metrics",
                      "P3_metrics_compare.png",
                      ["naive", "yes_no", "jaccard", "embedding"])
    for met in ("naive", "jaccard", "embedding"):
        _plot_grouped("P4_cross_change", met,
                      f"P4 — Cross-demographic answer change rate ({met} metric)",
                      f"P4_cross_{met}.png")
    _plot_p1_kinds("naive", "P1_kinds_naive.png")
    _plot_p1_kinds("jaccard", "P1_kinds_jaccard.png")
    _plot_p1_kinds("embedding", "P1_kinds_embedding.png")
    for ds in DATASETS:
        _plot_demographic("lenient", ds, f"P4_demo_{ds}.png")
        _plot_demographic("yes_no", ds, f"P4_demo_yesno_{ds}.png")
    print(f"wrote {len(list(PLOTS.glob('*.png')))} plots to {PLOTS}")


if __name__ == "__main__":
    main()
