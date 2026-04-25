"""P7 — Modality misattribution hallucination probe.

P7a: Direct modality recognition — \"What imaging modality is this?\"
     GT = ground-truth modality from sample's known modality.
P7b: Modality misattribution — phrase the question as if the image is a different modality.
     e.g. real image is MRI, ask: \"What abnormality is seen on this CT scan?\"
     The model should ideally either (1) refuse / correct, or (2) answer based on what's actually in the image.
     Hallucination = model accepts the false modality framing and produces an answer
     that explicitly references the wrong modality.

Outputs raw.jsonl with fields:
  sample_id, gt_modality, false_modality (or null for orig), question_form, pred, refusal, accepts_false_modality
"""
from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

REFUSAL_KW = ["cannot","can not","unable","not possible","unclear","not enough",
              "insufficient","need more","no image","unknown","i don","not sure",
              "actually","this is not","this image is not","not a ct","not an mri",
              "not a chest x-ray","not an x-ray","appears to be a","this is an mri",
              "this is a ct","this is an x-ray","this is an ultrasound"]

# Detect modality words appearing in the model's answer
MODALITY_TOKENS = {
    "ct":          re.compile(r"\b(ct\b|cta|computed tomograph)", re.I),
    "mri":         re.compile(r"\b(mri\b|mr\b|magnetic resonance|t1|t2|flair|dwi)", re.I),
    "x-ray":       re.compile(r"\b(x[-\s]?ray|xr\b|chest x|plain film|radiograph)", re.I),
    "ultrasound":  re.compile(r"\b(ultrasound|sonograph|doppler|us\b)", re.I),
    "angiography": re.compile(r"\b(angiograph|angiogram)\b", re.I),
}

# Misattribution probe questions (parameterized by false_modality)
MISATTR_QUESTIONS = [
    "What abnormality is seen on this {mod} image?",
    "What is the primary finding in this {mod} scan?",
    "Describe what you see in this {mod}.",
]

# Modality recognition question
RECOGNITION_QUESTION = "What imaging modality is this?"


def has_modality(text: str, modality: str) -> bool:
    pat = MODALITY_TOKENS.get(modality)
    if not pat: return False
    return bool(pat.search(text or ""))


def detected_modality(text: str) -> str | None:
    for mod, pat in MODALITY_TOKENS.items():
        if pat.search(text or ""): return mod
    return None


def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in REFUSAL_KW)


def expand_for_sample(sample, false_modalities):
    """Return list of (variant_id, image, question, meta) for one sample."""
    out = []
    out.append(("recog_orig", sample["image"], RECOGNITION_QUESTION,
                {"probe": "P7", "subprobe": "recognition", "kind": "orig"}))
    gt_mod = sample["modality"]
    for fm in false_modalities:
        if fm == gt_mod: continue  # skip true modality
        for i, qt in enumerate(MISATTR_QUESTIONS):
            out.append((f"misattr_{fm}_{i}", sample["image"],
                        qt.format(mod=fm),
                        {"probe": "P7", "subprobe": "misattribution",
                         "false_modality": fm, "kind": "wrong"}))
        # also one with the TRUE modality framing as control
        for i, qt in enumerate(MISATTR_QUESTIONS[:1]):
            out.append((f"misattr_true_{gt_mod}_{i}", sample["image"],
                        qt.format(mod=gt_mod),
                        {"probe": "P7", "subprobe": "misattribution",
                         "false_modality": gt_mod, "kind": "true"}))
            break  # only once
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llava_med", "biomed_clip"])
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets_loader import load
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "raw.jsonl"

    # Build pool of samples with KNOWN modality
    pool = []
    # VQA-Med 2019: sample_id with _modality suffix → GT itself is the modality
    samples = load("vqa_med_2019", 200)  # large pool
    for s in samples:
        if "_modality" in s["sample_id"]:
            mod = detected_modality(str(s["answer"]))
            if mod:
                s["modality"] = mod
                pool.append(s)

    # VQA-Med 2021 / VQA-RAD: detect modality from question + GT
    for ds_name in ["vqa_med_2021", "vqa_rad"]:
        more = load(ds_name, 200)
        for s in more:
            text = f"{s['answer']} {s['question']}"
            mod = detected_modality(text)
            if mod:
                s["modality"] = mod
                pool.append(s)

    # Cap to n_samples; balance across modalities if possible
    from collections import defaultdict
    by_mod = defaultdict(list)
    for s in pool: by_mod[s["modality"]].append(s)
    print(f"available pool by modality: {{m: len(v) for m, v in by_mod.items()}}".replace('{m: len(v) for m, v in by_mod.items()}', str({m: len(v) for m, v in by_mod.items()})), flush=True)
    selected = []
    per_mod = max(1, args.n_samples // len(by_mod))
    for mod, group in by_mod.items():
        selected.extend(group[:per_mod])
    selected = selected[:args.n_samples]
    print(f"running P7 on {len(selected)} samples", flush=True)

    # Build model
    if args.model == "llava_med":
        from models_wrapper import LlavaMedWrapper
        model = LlavaMedWrapper()
    else:
        from models_wrapper import BiomedClipWrapper
        model = BiomedClipWrapper()

    false_modalities = list(MODALITY_TOKENS.keys())

    f = open(out_jsonl, "w")
    import time
    ts = time.time()
    for si, s in enumerate(selected):
        variants = expand_for_sample(s, false_modalities)
        for vid, img, q, meta in variants:
            if args.model == "biomed_clip":
                cands = ["yes","no","cannot determine","ct","mri","x-ray","ultrasound",
                         "angiography","this is not a ct","this is not an mri",str(s["answer"])]
                pred = model.answer(img, q, candidates=list(dict.fromkeys(cands)))
            else:
                pred = model.answer(img, q)
            ans = pred.get("answer", "")
            rec = {
                "sample_id": s["sample_id"], "dataset": s.get("dataset"),
                "gt_modality": s["modality"], "gt_answer": s["answer"],
                "variant": vid, "question": q, "pred": ans,
                "refusal": is_refusal(ans),
                "mentions_gt_modality": has_modality(ans, s["modality"]),
                "mentions_false_modality": (
                    has_modality(ans, meta.get("false_modality"))
                    if meta.get("false_modality") and meta.get("kind") == "wrong" else None
                ),
                "detected_modality_in_pred": detected_modality(ans),
                "meta": meta,
            }
            f.write(json.dumps(rec, default=str) + "\n"); f.flush()
        if si % 5 == 0:
            print(f"[{si+1}/{len(selected)}] elapsed={time.time()-ts:.1f}s", flush=True)
    f.close()
    print(f"done. wrote {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
