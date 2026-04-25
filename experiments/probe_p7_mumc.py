"""P7 — Modality misattribution (MUMC-style).

MUMC's exact format:
  prefix = "This image was obtained using {modality}." + original question
  modalities tested = MRI, CT scan, X-ray, ultrasound, PET scan
  flip = pred(orig question) != pred(prefix + orig question)

This is fundamentally different from our previous P7 which used
"What abnormality is seen on this {mod} image?" — that REPLACED the
question, so flip was inflated by question difference.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# MUMC's 5 prefixes verbatim (from analysis_mumc/03_데이터셋별_결과.md)
MUMC_P7_PREFIXES = [
    "This image was obtained using MRI.",
    "This image was obtained using CT scan.",
    "This image was obtained using X-ray.",
    "This image was obtained using ultrasound.",
    "This image was obtained using PET scan.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llava_med", "biomed_clip"])
    ap.add_argument("--n_samples", type=int, default=120)
    ap.add_argument("--dataset", required=True, choices=["vqa_rad", "vqa_med_2019", "vqa_med_2021"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

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
        # original answer
        if args.model == "biomed_clip":
            cands = ["yes", "no", str(s["answer"]), "cannot determine", "normal", "abnormal"]
            cands = list(dict.fromkeys(cands))
            orig_pred = model.answer(s["image"], s["question"], candidates=cands)
        else:
            orig_pred = model.answer(s["image"], s["question"])
        orig_text = orig_pred.get("answer", "")
        # write orig as variant
        f.write(json.dumps({
            "sample_id": s["sample_id"], "dataset": args.dataset,
            "gt": s["answer"], "question": s["question"],
            "variant": "orig", "prefix_id": 0, "prefix_text": "",
            "pred": orig_text,
        }, default=str) + "\n")

        # 5 prefix variants
        for pi, pref in enumerate(MUMC_P7_PREFIXES, 1):
            q = f"{pref} {s['question']}"
            if args.model == "biomed_clip":
                pred = model.answer(s["image"], q, candidates=cands)
            else:
                pred = model.answer(s["image"], q)
            f.write(json.dumps({
                "sample_id": s["sample_id"], "dataset": args.dataset,
                "gt": s["answer"], "question": q,
                "variant": f"p7_{pi}", "prefix_id": pi, "prefix_text": pref,
                "orig_pred": orig_text, "pred": pred.get("answer", ""),
            }, default=str) + "\n")
        if si % 10 == 0:
            print(f"[{si+1}/{len(samples)}] elapsed={time.time()-ts:.1f}s", flush=True)
        f.flush()
    f.close()
    print(f"done. wrote {out_dir / 'raw.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
