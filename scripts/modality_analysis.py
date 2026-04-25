"""Modality-specific analysis.

VQA-Med 2019 has 4 explicit categories (C1=modality, C2=plane, C3=organ, C4=abnormality).
VQA-RAD doesn't have categories, but we can detect "modality questions" from question text.
VQA-Med 2021 is abnormality-only.

Goal: report per-category metrics so we can see, e.g., whether the model handles
modality questions differently from abnormality questions.

Outputs:
  results/modality/summary.csv          per-(model, dataset, category) headline metrics
  results/modality/plots/*.png          bar charts split by category
  results/modality/REPORT.md            한국어 자연어 리포트
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path
from collections import defaultdict, Counter
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from semantic_metrics import (
    normalize, contains_answer, jaccard, flip_naive, flip_jaccard, flip_yes_no,
    extract_yn,
)

OUT = ROOT / "results" / "modality"
OUT.mkdir(exist_ok=True)
PLOTS = OUT / "plots"; PLOTS.mkdir(exist_ok=True)

DATASETS = ["vqa_rad", "vqa_med_2019", "vqa_med_2021"]
MODELS = ["biomed_clip", "llava_med"]
COLORS = {"biomed_clip": "#4080c0", "llava_med": "#d04040"}

# VQA-RAD 카테고리 추론용 — 질문 text에서
RAD_MODALITY_Q = re.compile(r"\b(modality|imaging method|imaging technique|what kind of (?:image|scan|imaging)|what type of (?:image|scan|imaging)|is this a (?:ct|mri|x-?ray|ultrasound|us|pet))\b", re.I)
RAD_PLANE_Q = re.compile(r"\b(plane|axial|coronal|sagittal|longitudinal|transverse)\b", re.I)
RAD_ORGAN_Q = re.compile(r"\b(organ|what organ|which organ|organ system)\b", re.I)


def find_raw(model, dataset):
    for sub in ("_combined", "_big", "_full"):
        p = ROOT / "results" / f"{model}{sub}" / dataset / "raw.jsonl"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def category_for(sample_id: str, question: str, dataset: str) -> str:
    """Return one of: modality / plane / organ / abnormality / closed_yn / other"""
    if dataset == "vqa_med_2019":
        for c in ("modality", "plane", "organ", "abnormality"):
            if f"_{c}" in sample_id: return c
        return "other"
    if dataset == "vqa_med_2021":
        return "abnormality"
    if dataset == "vqa_rad":
        q = question or ""
        if RAD_MODALITY_Q.search(q): return "modality"
        if RAD_PLANE_Q.search(q): return "plane"
        if RAD_ORGAN_Q.search(q): return "organ"
        # closed yes/no
        ql = q.lower().strip()
        if ql.startswith(("is ", "are ", "does ", "do ", "was ", "were ", "has ", "have ")):
            return "closed_yn"
        return "abnormality"  # rough catch-all
    return "other"


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return p, max(0, center-half), min(1, center+half)


REFUSAL_KW = ["cannot","can not","unable","not possible","unclear","not enough",
              "insufficient","need more","no image","unknown","i don","not sure"]


def stats_per_category(recs):
    """Group records by category, then compute metrics per group."""
    by_cat = defaultdict(list)
    for r in recs:
        cat = category_for(r["sample_id"], r["question"], r["dataset"])
        r["category"] = cat
        by_cat[cat].append(r)

    out = {}
    for cat, rs in by_cat.items():
        # baseline (orig + P1)
        orig = [r for r in rs if r["probe"] == "P1_blank" and r["variant"] == "orig"]
        base_k = sum(contains_answer(r["pred"], r["gt"]) for r in orig)
        base_n = len(orig)

        # blank-image acc (does it still answer correctly without image?)
        blank_recs = [r for r in rs if r["probe"] == "P1_blank" and r["variant"] in ("black","white","gray","noise")]
        blank_k = sum(contains_answer(r["pred"], r["gt"]) for r in blank_recs)
        blank_n = len(blank_recs)

        # P1 flip rate (any blank vs orig)
        by_sample = defaultdict(dict)
        for r in rs:
            if r["probe"] != "P1_blank": continue
            by_sample[r["sample_id"]][r["variant"]] = r
        f1, t1 = 0, 0
        for vs in by_sample.values():
            if "orig" not in vs: continue
            for k in ("black","white","gray","noise"):
                if k not in vs: continue
                t1 += 1
                if normalize(vs["orig"]["pred"]) != normalize(vs[k]["pred"]):
                    f1 += 1

        # P2 mismatch hallucination
        p2_recs = [r for r in rs if r["probe"] == "P2_mismatch" and r["variant"] != "orig"]
        p2_n = len(p2_recs)
        p2_refuse = sum(any(kw in normalize(r["pred"]) for kw in REFUSAL_KW) for r in p2_recs)

        # P3 prefix flip
        by_s3 = defaultdict(dict)
        for r in rs:
            if r["probe"] != "P3_prefix": continue
            by_s3[r["sample_id"]][r["variant"]] = r
        f3, t3 = 0, 0
        for vs in by_s3.values():
            if "orig" not in vs: continue
            for v_id, r in vs.items():
                if v_id == "orig": continue
                t3 += 1
                if jaccard(vs["orig"]["pred"], r["pred"]) < 0.5:
                    f3 += 1

        # P4 cross-demographic change rate
        p4_recs = [r for r in rs if r["probe"] == "P4_demographic" and r["variant"] != "orig"]
        by_s4 = defaultdict(list)
        for r in p4_recs: by_s4[r["sample_id"]].append(r["pred"])
        cross_changes = []
        for sid, preds in by_s4.items():
            if len(preds) < 2: continue
            # jaccard-based change: pairs with sim < 0.5
            tot = 0; diff = 0
            for i in range(len(preds)):
                for j in range(i+1, len(preds)):
                    tot += 1
                    if jaccard(preds[i], preds[j]) < 0.5: diff += 1
            cross_changes.append(diff / max(1, tot))

        n_samples = len({r["sample_id"] for r in rs})
        out[cat] = {
            "n_samples": n_samples,
            "base_k": base_k, "base_n": base_n,
            "blank_k": blank_k, "blank_n": blank_n,
            "p1_f": f1, "p1_t": t1,
            "p2_refuse": p2_refuse, "p2_n": p2_n,
            "p3_f": f3, "p3_t": t3,
            "p4_cross_jaccard": float(np.mean(cross_changes)) if cross_changes else 0.0,
        }
    return out


def main():
    rows = []
    for m in MODELS:
        for d in DATASETS:
            p = find_raw(m, d)
            if not p: continue
            recs = [json.loads(l) for l in open(p)]
            print(f"=== {m}/{d}: {len(recs)} records, {len({r['sample_id'] for r in recs})} samples ===")
            cat_stats = stats_per_category(recs)
            for cat, s in cat_stats.items():
                base_p, base_lo, base_hi = wilson(s["base_k"], s["base_n"])
                blank_p, blank_lo, blank_hi = wilson(s["blank_k"], s["blank_n"])
                p1_p, p1_lo, p1_hi = wilson(s["p1_f"], s["p1_t"])
                p2_h_p, p2_h_lo, p2_h_hi = wilson(s["p2_n"] - s["p2_refuse"], s["p2_n"])
                p3_p, p3_lo, p3_hi = wilson(s["p3_f"], s["p3_t"])
                rows.append({
                    "model": m, "dataset": d, "category": cat,
                    "n_samples": s["n_samples"],
                    "baseline_acc": base_p, "baseline_lo": base_lo, "baseline_hi": base_hi,
                    "blank_acc": blank_p, "blank_lo": blank_lo, "blank_hi": blank_hi,
                    "P1_flip": p1_p, "P1_flip_lo": p1_lo, "P1_flip_hi": p1_hi,
                    "P2_halluc": p2_h_p, "P2_halluc_lo": p2_h_lo, "P2_halluc_hi": p2_h_hi,
                    "P3_flip_jaccard": p3_p, "P3_flip_lo": p3_lo, "P3_flip_hi": p3_hi,
                    "P4_cross_jaccard": s["p4_cross_jaccard"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    print(f"\nwrote {OUT / 'summary.csv'}  ({len(df)} rows)")
    # print main pivot
    pivot = df[["model","dataset","category","n_samples","baseline_acc","blank_acc","P1_flip","P2_halluc","P3_flip_jaccard"]]
    print(pivot.to_string(index=False))

    # Plots: per-metric, x=category, color=model, panel=dataset
    for metric, title in [
        ("baseline_acc", "Baseline accuracy (lenient)"),
        ("blank_acc",    "Blank-image accuracy"),
        ("P1_flip",      "P1 — answer flip rate (image blanked)"),
        ("P2_halluc",    "P2 — confident hallucination on out-of-scope"),
        ("P3_flip_jaccard", "P3 — irrelevant prefix flip (jaccard)"),
    ]:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 4), sharey=True)
        for ax, d in zip(axes, DATASETS):
            sub = df[df["dataset"] == d]
            cats = sorted(sub["category"].unique())
            x = np.arange(len(cats)); w = 0.38
            for i, m in enumerate(MODELS):
                rows = sub[sub["model"] == m].set_index("category").reindex(cats)
                vals = rows[metric].fillna(0).values
                ns = rows["n_samples"].fillna(0).astype(int).values
                ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=COLORS[m])
                for j, v in enumerate(vals):
                    if not pd.isna(v):
                        ax.text(x[j] + (i - 0.5) * w, v + 0.02, f"{v:.0%}\nn={ns[j]}", ha="center", fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9, rotation=20)
            ax.set_ylim(0, 1.1); ax.set_title(d)
        axes[0].set_ylabel(metric)
        axes[-1].legend()
        fig.suptitle(title)
        fig.tight_layout(); fig.savefig(PLOTS / f"{metric}.png", dpi=140); plt.close(fig)

    print(f"wrote {len(list(PLOTS.glob('*.png')))} plots to {PLOTS}")


if __name__ == "__main__":
    main()
