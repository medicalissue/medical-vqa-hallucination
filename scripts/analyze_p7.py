"""Analyze P7 modality misattribution results.

For each model:
  - P7a recognition accuracy: how often does the model correctly identify modality?
  - P7b acceptance rate: when given a false-modality prompt, does the model
    (a) refuse / correct the modality
    (b) explicitly mention the FALSE modality in its answer (worst case — hallucinates the framing)
    (c) just answer based on what's actually in the image (mentions GT modality)

Also: cross-modality confusion matrix — what does the model actually say for each true modality?
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict, Counter
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "p7_analysis"
OUT.mkdir(exist_ok=True)
PLOTS = OUT / "plots"; PLOTS.mkdir(exist_ok=True)

MODALITIES = ["x-ray", "ct", "mri", "ultrasound", "angiography"]
MODELS = ["biomed_clip", "llava_med"]
COLORS = {"biomed_clip": "#4080c0", "llava_med": "#d04040"}


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return p, max(0, center-half), min(1, center+half)


def analyze(model_name: str):
    # prefer _big over base
    for suffix in ("_big", ""):
        p = ROOT / "results" / f"p7_{model_name}{suffix}" / "raw.jsonl"
        if p.exists() and p.stat().st_size > 0:
            break
    if not p.exists():
        print(f"[!] {p} not found")
        return None
    recs = [json.loads(l) for l in open(p)]
    print(f"[{model_name}] {len(recs)} records")

    # P7a recognition accuracy
    recog = [r for r in recs if r["meta"]["subprobe"] == "recognition"]
    recog_correct = sum(r["mentions_gt_modality"] for r in recog)

    # P7b: misattribution
    mis_wrong = [r for r in recs if r["meta"].get("subprobe") == "misattribution" and r["meta"].get("kind") == "wrong"]
    mis_true = [r for r in recs if r["meta"].get("subprobe") == "misattribution" and r["meta"].get("kind") == "true"]

    # in mis_wrong: how often does the model mention the FALSE modality?
    mentions_false = sum(bool(r["mentions_false_modality"]) for r in mis_wrong)
    refusal_or_correction = sum(r["refusal"] for r in mis_wrong)

    # confusion: for each true modality, what does model say (recognition probe)?
    confusion = defaultdict(Counter)
    for r in recog:
        gt = r["gt_modality"]
        det = r["detected_modality_in_pred"] or "none"
        confusion[gt][det] += 1

    # per-(true_mod, false_mod) — when prompted with false modality, how often is it mentioned in pred?
    per_pair = defaultdict(lambda: [0, 0])  # mentions / total
    for r in mis_wrong:
        true_mod = r["gt_modality"]
        fmod = r["meta"]["false_modality"]
        per_pair[(true_mod, fmod)][1] += 1
        if r["mentions_false_modality"]:
            per_pair[(true_mod, fmod)][0] += 1

    summary = {
        "model": model_name,
        "P7a_recognition_acc": (recog_correct, len(recog)),
        "P7b_total_misattr": len(mis_wrong),
        "P7b_refusal_rate": (refusal_or_correction, len(mis_wrong)),
        "P7b_mentions_false_modality_rate": (mentions_false, len(mis_wrong)),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "per_pair": {f"{a}->{b}": v for (a, b), v in per_pair.items()},
    }
    return summary, recs


def main():
    summaries = {}
    all_recs = {}
    for m in MODELS:
        out = analyze(m)
        if out:
            summaries[m], all_recs[m] = out

    # Print summary
    rows = []
    for m, s in summaries.items():
        rec_p, rec_lo, rec_hi = wilson(*s["P7a_recognition_acc"])
        ref_p, ref_lo, ref_hi = wilson(*s["P7b_refusal_rate"])
        mf_p, mf_lo, mf_hi = wilson(*s["P7b_mentions_false_modality_rate"])
        rows.append({
            "model": m,
            "P7a_recog_acc": f"{rec_p:.1%} [{rec_lo:.1%}, {rec_hi:.1%}]",
            "P7a_n": s["P7a_recognition_acc"][1],
            "P7b_refusal": f"{ref_p:.1%} [{ref_lo:.1%}, {ref_hi:.1%}]",
            "P7b_mentions_false_mod": f"{mf_p:.1%} [{mf_lo:.1%}, {mf_hi:.1%}]",
            "P7b_n": s["P7b_total_misattr"],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "summary.csv", index=False)

    # Plot 1: P7a recognition accuracy + P7b refusal vs hallucination
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (col_key, title) in zip(axes, [
        ("P7a_recognition_acc", "P7a — Modality recognition acc"),
        ("P7b_refusal_rate", "P7b — Refusal/correction rate (높을수록 좋음)"),
        ("P7b_mentions_false_modality_rate", "P7b — Pred에 wrong modality 언급 (낮을수록 좋음)"),
    ]):
        for i, m in enumerate(MODELS):
            if m not in summaries: continue
            k, n = summaries[m][col_key]
            p, lo, hi = wilson(k, n)
            ax.bar([m], [p], color=COLORS[m],
                   yerr=[[max(0, p-lo)], [max(0, hi-p)]], capsize=4)
            ax.text(i, p + 0.02, f"{p:.0%}\n(n={n})", ha="center")
        ax.set_ylim(0, 1.05); ax.set_title(title)
    axes[0].set_ylabel("rate")
    fig.tight_layout(); fig.savefig(PLOTS / "p7_summary.png", dpi=140); plt.close(fig)

    # Plot 2: confusion matrix (true mod x detected mod) for each model
    for m, s in summaries.items():
        conf = s["confusion"]
        all_dets = sorted({d for v in conf.values() for d in v} | set(MODALITIES))
        gts = sorted(conf.keys())
        mat = np.zeros((len(gts), len(all_dets)))
        for i, gt in enumerate(gts):
            for j, det in enumerate(all_dets):
                mat[i, j] = conf[gt].get(det, 0)
        # row-normalize
        row_sum = mat.sum(axis=1, keepdims=True); row_sum[row_sum==0] = 1
        mat_norm = mat / row_sum
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(all_dets))); ax.set_xticklabels(all_dets, rotation=30, ha="right")
        ax.set_yticks(range(len(gts))); ax.set_yticklabels(gts)
        for i in range(len(gts)):
            for j in range(len(all_dets)):
                if mat[i,j] > 0:
                    ax.text(j, i, f"{int(mat[i,j])}\n{mat_norm[i,j]:.0%}", ha="center", va="center",
                           color="white" if mat_norm[i,j] > 0.5 else "black", fontsize=8)
        ax.set_xlabel("detected modality in prediction")
        ax.set_ylabel("ground truth modality")
        ax.set_title(f"P7a — Recognition confusion ({m})")
        fig.colorbar(im, ax=ax, label="row-normalized")
        fig.tight_layout(); fig.savefig(PLOTS / f"p7_confusion_{m}.png", dpi=140); plt.close(fig)

    # Plot 3: per (true, false) modality pair acceptance rate (LLaVA only — generative)
    if "llava_med" in summaries:
        per_pair = summaries["llava_med"]["per_pair"]
        # assemble matrix: rows = true_mod, cols = false_mod
        mat = np.zeros((len(MODALITIES), len(MODALITIES)))
        n_mat = np.zeros_like(mat, dtype=int)
        for key, (k, n) in per_pair.items():
            tm, fm = key.split("->")
            if tm in MODALITIES and fm in MODALITIES:
                i, j = MODALITIES.index(tm), MODALITIES.index(fm)
                mat[i, j] = k / n if n else np.nan
                n_mat[i, j] = n
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(len(MODALITIES))); ax.set_xticklabels(MODALITIES, rotation=30)
        ax.set_yticks(range(len(MODALITIES))); ax.set_yticklabels(MODALITIES)
        for i in range(len(MODALITIES)):
            for j in range(len(MODALITIES)):
                if n_mat[i,j] > 0:
                    ax.text(j, i, f"{mat[i,j]:.0%}\n(n={n_mat[i,j]})", ha="center", va="center",
                           color="white" if mat[i,j] > 0.5 else "black", fontsize=8)
                elif i == j:
                    ax.text(j, i, "—", ha="center", va="center", color="gray")
        ax.set_xlabel("FALSE modality injected in prompt")
        ax.set_ylabel("TRUE image modality")
        ax.set_title("P7b — LLaVA-Med: rate of mentioning FALSE modality\n(높을수록 hallucination 심함)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(PLOTS / "p7_misattr_matrix_llava.png", dpi=140); plt.close(fig)

    # Save full summaries.json
    with open(OUT / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
