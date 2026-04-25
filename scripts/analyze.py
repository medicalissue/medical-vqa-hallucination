"""Generate plots and a markdown summary from raw.jsonl files.

Reads results/<model>/raw.jsonl and writes results/<model>/plots/*.png + report.md
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from metrics import accuracy, refusal_rate, kl_to_reference, demographic_disparity


def load(jsonl_path: Path) -> list[dict]:
    return [json.loads(l) for l in open(jsonl_path)]


def plot_blank_flip(records, out_path: Path):
    by_sample = defaultdict(dict)
    for r in records:
        if r["probe"] != "P1_blank": continue
        by_sample[r["sample_id"]][r["variant"]] = r["pred"]
    kinds = ["black", "white", "noise", "gray"]
    flip_rates = []
    for k in kinds:
        n_total = n_flip = 0
        for sid, vs in by_sample.items():
            if "orig" in vs and k in vs:
                n_total += 1
                if vs[k] != vs["orig"]: n_flip += 1
        flip_rates.append(n_flip / max(1, n_total))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(kinds, flip_rates, color=["#222", "#ddd", "#888", "#555"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Answer flip rate (vs original image)")
    ax.set_title("P1 — Blank Image: how often does the answer change?\n(higher = model relies on image)")
    for i, v in enumerate(flip_rates):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_demographic_acc(records, out_path: Path):
    p4 = [r for r in records if r["probe"] == "P4_demographic" and r["variant"] != "orig"]
    if not p4: return
    by_demo = defaultdict(list)
    for r in p4:
        by_demo[r["meta"]["demo"]].append((r["pred"], r["gt"]))
    rows = sorted(((d, accuracy([p for p,_ in v], [g for _,g in v])) for d, v in by_demo.items()),
                  key=lambda x: x[0])
    demos, accs = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(demos, accs, color="#4080c0")
    ax.set_xticklabels(demos, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("P4 — Demographic prefix: per-group accuracy\n(gap reveals hidden bias from prepended text)")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_probe_summary(records, out_path: Path):
    """Stacked bar: refusal vs hallucination rates per probe (P1/P2/P3)."""
    metrics = {}
    # P2 mismatch
    p2 = [r for r in records if r["probe"] == "P2_mismatch" and r["variant"] != "orig"]
    if p2:
        rr = refusal_rate([r["pred"] for r in p2])
        metrics["P2_mismatch"] = (rr, 1 - rr)
    # P3 prefix flip rate
    p3 = [r for r in records if r["probe"] == "P3_prefix"]
    if p3:
        by_sample = defaultdict(dict)
        for r in p3: by_sample[r["sample_id"]][r["variant"]] = r["pred"]
        flips = []
        for sid, vs in by_sample.items():
            orig = vs.get("orig")
            for k, v in vs.items():
                if k == "orig" or orig is None: continue
                flips.append(v != orig)
        metrics["P3_prefix_flip"] = (sum(flips)/len(flips), 1 - sum(flips)/len(flips)) if flips else (0, 1)

    if not metrics: return
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = list(metrics.keys())
    a = [metrics[k][0] for k in keys]; b = [metrics[k][1] for k in keys]
    ax.barh(keys, a, color="#4080c0", label="refusal / flip")
    ax.barh(keys, b, left=a, color="#d04040", label="answered / unchanged")
    ax.set_xlim(0, 1); ax.legend()
    ax.set_title("Probe overview: refusal vs answered, flip vs unchanged")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def write_report(records, model_name: str, out_md: Path, plots_dir: Path):
    n = len(records)
    n_samples = len({r["sample_id"] for r in records})
    by_probe = Counter(r["probe"] for r in records)
    md = [f"# {model_name} — Hallucination Probe Report\n",
          f"- records: **{n}**  / unique samples: **{n_samples}**",
          f"- probes: {dict(by_probe)}",
          ""]
    md.append("## Plots\n")
    for png in sorted(plots_dir.glob("*.png")):
        md.append(f"![{png.stem}]({png.name})")
        md.append("")
    md.append("## Example hallucinations (P2 image-text mismatch)\n")
    p2 = [r for r in records if r["probe"] == "P2_mismatch" and r["variant"] != "orig"][:6]
    for r in p2:
        md.append(f"- *Q*: {r['question']}\n  *Pred*: `{r['pred']}`  (GT for image: `{r['gt']}`)")
    md.append("")
    md.append("## Demographic-flip examples (P4)\n")
    p4 = [r for r in records if r["probe"] == "P4_demographic" and r["variant"] != "orig"]
    by_sample = defaultdict(dict)
    for r in p4: by_sample[r["sample_id"]][r["meta"].get("demo")] = r["pred"]
    flipped = [(sid, demos) for sid, demos in by_sample.items()
               if len({str(v).strip().lower() for v in demos.values()}) > 1]
    for sid, demos in flipped[:5]:
        md.append(f"- sample `{sid}`:")
        for d, p in demos.items(): md.append(f"  - {d}: `{p}`")
        md.append("")
    out_md.write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="path to raw.jsonl")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"; plots.mkdir(exist_ok=True)
    recs = load(Path(args.raw))
    plot_blank_flip(recs, plots / "p1_blank_flip.png")
    plot_demographic_acc(recs, plots / "p4_demographic_acc.png")
    plot_probe_summary(recs, plots / "probe_overview.png")
    write_report(recs, args.model_name, out / "report.md", plots)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
