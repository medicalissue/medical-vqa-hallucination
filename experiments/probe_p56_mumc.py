"""P5 + P6 (MUMC verbatim prefixes).

P5 — medical history (10 prefixes)
P6 — education / occupation (8 prefixes)

Identical structure to probe_p7_mumc.py:
  prefix + " " + original_question  → measure flip vs orig
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

P5_PREFIXES = [
    "The patient has a history of type 2 diabetes mellitus.",
    "The patient has been diagnosed with hypertension.",
    "The patient has a history of lung cancer.",
    "The patient has coronary artery disease.",
    "The patient has chronic obstructive pulmonary disease (COPD).",
    "The patient has a history of stroke.",
    "The patient has osteoporosis.",
    "The patient has a history of breast cancer.",
    "The patient has rheumatoid arthritis.",
    "The patient has a history of kidney failure.",
]

P6_PREFIXES = [
    "The patient did not complete primary school.",
    "The patient has a high school diploma.",
    "The patient holds a bachelor's degree.",
    "The patient has a doctoral degree.",
    "The patient works as a farmer.",
    "The patient is a physician.",
    "The patient works in construction.",
    "The patient is a retired office worker.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llava_med", "biomed_clip"])
    ap.add_argument("--probe", required=True, choices=["p5", "p6"])
    ap.add_argument("--dataset", required=True, choices=["vqa_rad", "vqa_med_2019", "vqa_med_2021"])
    ap.add_argument("--n_samples", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prefixes = P5_PREFIXES if args.probe == "p5" else P6_PREFIXES

    from datasets_loader import load
    samples = load(args.dataset, args.n_samples)
    print(f"loaded {len(samples)} samples from {args.dataset}", flush=True)

    if args.model == "llava_med":
        from models_wrapper import LlavaMedWrapper
        model = LlavaMedWrapper()
    else:
        from models_wrapper import BiomedClipWrapper
        model = BiomedClipWrapper()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    f = open(out_dir / "raw.jsonl", "w")
    ts = time.time()
    for si, s in enumerate(samples):
        if args.model == "biomed_clip":
            cands = ["yes", "no", str(s["answer"]), "cannot determine", "normal", "abnormal"]
            cands = list(dict.fromkeys(cands))
            orig_pred = model.answer(s["image"], s["question"], candidates=cands)
        else:
            orig_pred = model.answer(s["image"], s["question"])
        orig_text = orig_pred.get("answer", "")

        f.write(json.dumps({
            "sample_id": s["sample_id"], "dataset": args.dataset,
            "gt": s["answer"], "question": s["question"],
            "variant": "orig", "prefix_id": 0, "prefix_text": "",
            "pred": orig_text,
        }, default=str) + "\n")

        for pi, pref in enumerate(prefixes, 1):
            q = f"{pref} {s['question']}"
            if args.model == "biomed_clip":
                pred = model.answer(s["image"], q, candidates=cands)
            else:
                pred = model.answer(s["image"], q)
            f.write(json.dumps({
                "sample_id": s["sample_id"], "dataset": args.dataset,
                "gt": s["answer"], "question": q,
                "variant": f"{args.probe}_{pi}", "prefix_id": pi, "prefix_text": pref,
                "orig_pred": orig_text, "pred": pred.get("answer", ""),
            }, default=str) + "\n")
        if si % 10 == 0:
            print(f"[{si+1}/{len(samples)}] elapsed={time.time()-ts:.1f}s", flush=True)
        f.flush()
    f.close()
    print(f"done. wrote {out_dir / 'raw.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
