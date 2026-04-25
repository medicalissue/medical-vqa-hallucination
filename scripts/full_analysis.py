"""Aggregate cross-dataset, cross-model results into a unified DataFrame and plots.

Reads:
    results/biomed_clip_full/<ds>/raw.jsonl
    results/llava_med_full/<ds>/raw.jsonl

Writes:
    results/full/summary.csv
    results/full/plots/*.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from metrics import accuracy, refusal_rate, _norm, _contains_answer, kl_to_reference

DATASETS = ["vqa_rad", "vqa_med_2019", "vqa_med_2021"]
MODELS = ["biomed_clip", "llava_med"]


def load(model: str, dataset: str):
    p = ROOT / "results" / f"{model}_full" / dataset / "raw.jsonl"
    if not p.exists(): return None
    return [json.loads(l) for l in open(p)]


def stats(recs):
    by_probe = defaultdict(list)
    for r in recs: by_probe[r["probe"]].append(r)
    n_samples = len({r["sample_id"] for r in recs})
    orig = [r for r in recs if r["probe"] == "P1_blank" and r["variant"] == "orig"]
    base = accuracy([r["pred"] for r in orig], [r["gt"] for r in orig])
    blank_recs = [r for r in by_probe["P1_blank"] if r["variant"] in ("black", "white", "noise", "gray")]
    blank_acc = accuracy([r["pred"] for r in blank_recs], [r["gt"] for r in blank_recs])
    by_sample = defaultdict(dict)
    for r in by_probe["P1_blank"]: by_sample[r["sample_id"]][r["variant"]] = r["pred"]
    p1_flips = []
    for vs in by_sample.values():
        for k in ("black", "white", "noise", "gray"):
            if k in vs and "orig" in vs:
                p1_flips.append(_norm(vs[k]) != _norm(vs["orig"]))
    p1_flip = np.mean(p1_flips) if p1_flips else 0
    p2_preds = [r["pred"] for r in by_probe["P2_mismatch"] if r["variant"] != "orig"]
    p2_halluc = 1 - refusal_rate(p2_preds)
    by_s3 = defaultdict(dict)
    for r in by_probe["P3_prefix"]: by_s3[r["sample_id"]][r["variant"]] = r["pred"]
    p3_flips = []
    for vs in by_s3.values():
        for k, v in vs.items():
            if k == "orig": continue
            if "orig" in vs: p3_flips.append(_norm(v) != _norm(vs["orig"]))
    p3_flip = np.mean(p3_flips) if p3_flips else 0
    p4_recs = [r for r in by_probe["P4_demographic"] if r["variant"] != "orig"]
    by_demo = defaultdict(list)
    for r in p4_recs: by_demo[r["meta"]["demo"]].append((r["pred"], r["gt"]))
    accs = [accuracy([p for p, _ in v], [g for _, g in v]) for v in by_demo.values()]
    p4_gap = max(accs) - min(accs) if accs else 0
    by_s4 = defaultdict(list)
    for r in p4_recs: by_s4[r["sample_id"]].append(r["pred"])
    cross_changes = []
    for sid, ps in by_s4.items():
        if len(ps) < 2: continue
        u = len({_norm(p) for p in ps})
        cross_changes.append((u - 1) / (len(ps) - 1))
    p4_change = np.mean(cross_changes) if cross_changes else 0
    return {
        "n_samples": n_samples, "n_records": len(recs),
        "baseline_acc": base, "blank_acc": blank_acc,
        "P1_flip": p1_flip, "P2_halluc": p2_halluc,
        "P3_flip": p3_flip, "P4_max_gap": p4_gap, "P4_cross_change": p4_change,
    }


def main():
    rows = []
    for m in MODELS:
        for d in DATASETS:
            recs = load(m, d)
            if recs is None: continue
            s = stats(recs)
            s["model"] = m; s["dataset"] = d
            rows.append(s)
    df = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(df.to_string(index=False))

    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)

    # Plot 1: per-metric heatmap-style grouped bar (model × dataset)
    metrics_to_plot = [
        ("baseline_acc", "Baseline accuracy (lenient)"),
        ("blank_acc", "Accuracy on BLANK image"),
        ("P1_flip", "P1 — answer flip rate (image blanked)"),
        ("P2_halluc", "P2 — confident hallucination rate"),
        ("P3_flip", "P3 — prefix flip rate"),
        ("P4_max_gap", "P4 — demographic max accuracy gap"),
        ("P4_cross_change", "P4 — cross-demographic change rate"),
    ]
    for col, title in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(DATASETS)); w = 0.38
        for i, m in enumerate(MODELS):
            sub = df[df["model"] == m].set_index("dataset").reindex(DATASETS)
            vals = sub[col].fillna(0).values
            ax.bar(x + (i - 0.5) * w, vals, w,
                   label=m, color="#4080c0" if m == "biomed_clip" else "#d04040")
            for j, v in enumerate(vals):
                ax.text(x[j] + (i - 0.5) * w, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(DATASETS); ax.set_ylim(0, 1.05)
        ax.set_ylabel("rate / accuracy"); ax.legend(); ax.set_title(title)
        fig.tight_layout(); fig.savefig(plots / f"{col}.png", dpi=140); plt.close(fig)

    # Plot 2: side-by-side per-dataset radar-ish bar
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 4), sharey=True)
    for ax, d in zip(axes, DATASETS):
        sub = df[df["dataset"] == d].set_index("model")
        cols = ["P1_flip", "P2_halluc", "P3_flip", "P4_max_gap", "P4_cross_change"]
        x = np.arange(len(cols)); w = 0.38
        for i, m in enumerate(MODELS):
            if m not in sub.index: continue
            vals = [sub.loc[m, c] for c in cols]
            ax.bar(x + (i - 0.5) * w, vals, w, label=m,
                   color="#4080c0" if m == "biomed_clip" else "#d04040")
        ax.set_xticks(x); ax.set_xticklabels([c[:8] for c in cols], fontsize=8, rotation=30)
        ax.set_ylim(0, 1.05); ax.set_title(d)
    axes[0].set_ylabel("rate / accuracy")
    axes[-1].legend()
    fig.suptitle("Probe profiles per dataset")
    fig.tight_layout(); fig.savefig(plots / "per_dataset_profile.png", dpi=140); plt.close(fig)

    print(f"wrote {out_dir} with {len(rows)} rows")


if __name__ == "__main__":
    main()
