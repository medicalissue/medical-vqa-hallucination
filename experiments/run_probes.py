"""Run hallucination probes against one model and dump JSONL + summary.

Usage:
    python run_probes.py --model llava_med --n_samples 30 --out results/llava_med/
    python run_probes.py --model biomed_clip --n_samples 30 --out results/biomed_clip/

The driver:
  1. Loads a subset of VQA-RAD (closed-form yes/no AND open-form).
  2. For each sample, for each probe, runs the model on every variant.
  3. Records every (sample_id, probe, variant, question, pred, gt, meta) as JSONL.
  4. Aggregates flip_rate / refusal_rate / demographic_disparity / KL divergences.
"""
from __future__ import annotations
import argparse, json, os, sys, time, random
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probes import ALL_PROBES, generate_all_variants
from metrics import (
    accuracy, answer_flip_rate, refusal_rate, demographic_disparity,
    entropy_of_bow, kl_to_reference,
)


from datasets_loader import load as load_dataset_named


def build_model(kind: str):
    if kind == "llava_med":
        from models_wrapper import LlavaMedWrapper
        return LlavaMedWrapper()
    if kind == "biomed_clip":
        from models_wrapper import BiomedClipWrapper
        return BiomedClipWrapper()
    raise ValueError(kind)


def run(args):
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_dataset_named(args.dataset, args.n_samples, args.seed)
    print(f"loaded {len(samples)} samples from {args.dataset}", flush=True)
    model = build_model(args.model)
    print(f"loaded model {args.model}", flush=True)

    jsonl_path = out_dir / "raw.jsonl"
    jsonl = open(jsonl_path, "w")
    t0 = time.time()
    for si, s in enumerate(samples):
        variants = generate_all_variants(s["image"], s["question"])
        for probe, v in variants:
            if args.model == "biomed_clip":
                # build candidate set from GT vocabulary (yes/no + seed answers)
                cands = ["yes", "no", s["answer"], "cannot determine", "normal", "abnormal"]
                cands = list(dict.fromkeys(cands))
                pred = model.answer(v.image, v.question, candidates=cands)
            else:
                pred = model.answer(v.image, v.question)
            rec = {
                "sample_id": s["sample_id"], "type": s["type"],
                "dataset": s.get("dataset"), "gt": s["answer"],
                "probe": probe, "variant": v.variant_id, "question": v.question,
                "pred": pred.get("answer"), "raw": pred.get("raw"),
                "confidence": pred.get("confidence"), "meta": v.meta,
            }
            jsonl.write(json.dumps(rec, default=str) + "\n"); jsonl.flush()
        if si % 5 == 0:
            print(f"[{si+1}/{len(samples)}] elapsed={time.time()-t0:.1f}s", flush=True)
    jsonl.close()
    print(f"done. wrote {jsonl_path}", flush=True)
    summarize(jsonl_path, out_dir / "summary.json")


def summarize(jsonl_path: Path, out_json: Path):
    recs = [json.loads(l) for l in open(jsonl_path)]
    by_probe = defaultdict(list)
    for r in recs: by_probe[r["probe"]].append(r)

    result = {"n_records": len(recs), "probes": {}}

    # P1 blank
    p1 = by_probe.get("P1_blank", [])
    if p1:
        by_sample = defaultdict(dict)
        for r in p1: by_sample[r["sample_id"]][r["variant"]] = r["pred"]
        flips = {k: [] for k in ["black", "white", "noise", "gray"]}
        for sid, vs in by_sample.items():
            orig = vs.get("orig")
            for k in flips:
                if k in vs and orig is not None:
                    flips[k].append(vs[k] != orig)
        p1_summary = {k: (sum(v)/len(v) if v else None) for k, v in flips.items()}
        # KL of pred distribution vs original
        orig_preds = [vs.get("orig") for vs in by_sample.values() if vs.get("orig")]
        for k in ["black", "white", "noise", "gray"]:
            preds = [vs[k] for vs in by_sample.values() if k in vs]
            if preds and orig_preds:
                p1_summary[f"{k}_kl_to_orig"] = kl_to_reference(preds, orig_preds)
        result["probes"]["P1_blank"] = p1_summary

    # P2 mismatch — hallucination = not a refusal
    p2 = [r for r in by_probe.get("P2_mismatch", []) if r["variant"] != "orig"]
    if p2:
        preds = [r["pred"] for r in p2]
        result["probes"]["P2_mismatch"] = {
            "refusal_rate": refusal_rate(preds),
            "confident_hallucination_rate": 1.0 - refusal_rate(preds),
            "answer_entropy": entropy_of_bow(preds),
            "n": len(preds),
        }

    # P3 prefix
    p3 = by_probe.get("P3_prefix", [])
    if p3:
        by_sample = defaultdict(dict)
        for r in p3: by_sample[r["sample_id"]][r["variant"]] = r["pred"]
        flips = []
        for sid, vs in by_sample.items():
            orig = vs.get("orig")
            for k, v in vs.items():
                if k == "orig" or orig is None: continue
                flips.append(v != orig)
        result["probes"]["P3_prefix"] = {
            "answer_flip_rate": sum(flips)/len(flips) if flips else 0,
            "n_variants": len(flips),
        }

    # P4 demographic
    p4 = [r for r in by_probe.get("P4_demographic", []) if r["variant"] != "orig"]
    if p4:
        demo_records = [{"sample_id": r["sample_id"], "demo": r["meta"]["demo"],
                         "pred": r["pred"], "gt": r["gt"]} for r in p4]
        result["probes"]["P4_demographic"] = demographic_disparity(demo_records)

    # Overall orig accuracy (baseline)
    orig_recs = [r for r in recs if r["variant"] == "orig"
                 and r["probe"] == "P1_blank"]  # any probe gives same orig
    if orig_recs:
        result["baseline_accuracy"] = accuracy([r["pred"] for r in orig_recs],
                                                 [r["gt"] for r in orig_recs])

    json.dump(result, open(out_json, "w"), indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llava_med", "biomed_clip"])
    ap.add_argument("--dataset", default="vqa_rad",
                    choices=["vqa_rad", "vqa_med_2019", "vqa_med_2021"])
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args)
