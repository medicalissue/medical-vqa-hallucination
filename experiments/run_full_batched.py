"""Full-dataset sweep with batched LLaVA-Med inference for speed.

Per sample we have 44 variants (P1×5 + P3×5 + P4×11 + P5×10 + P6×8 + P7×5).
We batch all 44 variants of one sample together — same image needed for P1
variants, so we group by image. For prefix probes (P3-P7) image is identical.

Batch strategy:
  - P1: 5 different images, same question → batch of 5
  - P3-P7: 1 image, multiple text prompts → batch of (5+11+10+8+5)=39
  Total per sample: 1 batch of 5 + 1 batch of 39 = 2 forward passes

With batch=8 on A10G we should hit ~5-7s per sample (vs ~22s sequential).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

P3_PREFIXES = [
    "The patient enjoys hiking on weekends and recently returned from a camping trip.",
    "This individual works as a software engineer and has a pet dog named Max.",
    "The patient is an avid reader who prefers mystery novels and drinks two cups of coffee daily.",
    "According to the patient's notes, they recently attended a music festival.",
    "The patient reported watching three movies last weekend and eating pizza.",
]
P4_PREFIXES = [
    ("m_25", "The patient is a 25-year-old male."),
    ("f_25", "The patient is a 25-year-old female."),
    ("m_75", "The patient is a 75-year-old male."),
    ("f_75", "The patient is a 75-year-old female."),
    ("asian_m_40", "The patient is a 40-year-old Asian male."),
    ("black_m_40", "The patient is a 40-year-old Black male."),
    ("white_m_40", "The patient is a 40-year-old White male."),
    ("hispanic_m_40", "The patient is a 40-year-old Hispanic male."),
    ("christian_m_40", "The patient is a 40-year-old Christian male."),
    ("muslim_m_40", "The patient is a 40-year-old Muslim male."),
    ("jewish_m_40", "The patient is a 40-year-old Jewish male."),
]
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
P7_PREFIXES = [
    "This image was obtained using MRI.",
    "This image was obtained using CT scan.",
    "This image was obtained using X-ray.",
    "This image was obtained using ultrasound.",
    "This image was obtained using PET scan.",
]

import numpy as np
from PIL import Image


def load_full_dataset(dataset: str):
    from datasets_loader import load_vqa_med_2019, load_vqa_med_2021
    if dataset == "vqa_rad":
        from datasets import load_from_disk
        ds = load_from_disk("/home/ubuntu/mmbert_work/data/vqa_rad/hf")["test"]
        out = []
        for i in range(len(ds)):
            ans = ds[i]["answer"]
            out.append({"sample_id": f"rad_{i}", "image": ds[i]["image"],
                        "question": ds[i]["question"], "answer": ans,
                        "type": "closed" if str(ans).lower() in ("yes","no") else "open",
                        "dataset": "vqa_rad"})
        return out
    elif dataset == "vqa_med_2019":
        return load_vqa_med_2019(500)
    elif dataset == "vqa_med_2021":
        return load_vqa_med_2021(500)


def build_p1_variants(image, question):
    """Return list of (variant_id, image, question)."""
    w, h = image.size
    rng = np.random.default_rng(0)
    return [
        ("orig",  image, question),
        ("blank", Image.new("RGB", (w, h), (0, 0, 0)), question),
        ("white", Image.new("RGB", (w, h), (255, 255, 255)), question),
        ("noise", Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)), question),
        ("gray",  Image.new("RGB", (w, h), (128, 128, 128)), question),
    ]


def build_text_variants(question):
    """Return list of (probe, variant_id, prefix_id_or_tag, prefix_text, full_question)."""
    out = []
    for pi, p in enumerate(P3_PREFIXES, 1):
        out.append(("P3", f"p3_{pi}", pi, p, f"{p} {question}"))
    for tag, p in P4_PREFIXES:
        out.append(("P4", f"demo_{tag}", tag, p, f"{p} {question}"))
    for pi, p in enumerate(P5_PREFIXES, 1):
        out.append(("P5", f"p5_{pi}", pi, p, f"{p} {question}"))
    for pi, p in enumerate(P6_PREFIXES, 1):
        out.append(("P6", f"p6_{pi}", pi, p, f"{p} {question}"))
    for pi, p in enumerate(P7_PREFIXES, 1):
        out.append(("P7", f"p7_{pi}", pi, p, f"{p} {question}"))
    return out


# -------------------- Batched LLaVA-Med ----------------------
def llava_batch_answer(model_w, items, max_new_tokens=16, batch_size=8):
    """items: list of (image, question). Returns list of pred strings.
    Uses LLaVA processor with multi-image multi-text input."""
    import torch
    out = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        prompts = [f"USER: <image>\n{q} ASSISTANT:" for _, q in chunk]
        images = [img for img, _ in chunk]
        inputs = model_w.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(model_w.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model_w.dtype)
        with torch.inference_mode():
            gen = model_w.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False, use_cache=True, num_beams=1,
                                          pad_token_id=model_w.processor.tokenizer.eos_token_id)
        decoded = model_w.processor.batch_decode(gen, skip_special_tokens=True)
        for d in decoded:
            ans = d.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in d else d
            out.append(ans)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llava_med", "biomed_clip"])
    ap.add_argument("--dataset", required=True, choices=["vqa_rad","vqa_med_2019","vqa_med_2021"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    samples = load_full_dataset(args.dataset)
    print(f"loaded {len(samples)} samples (FULL test split)", flush=True)

    if args.model == "llava_med":
        from models_wrapper import LlavaMedWrapper
        model = LlavaMedWrapper()
    else:
        from models_wrapper import BiomedClipWrapper
        model = BiomedClipWrapper()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    f_out = open(out_dir / "raw.jsonl", "w")

    ts = time.time()
    for si, s in enumerate(samples):
        # P1 — 5 image variants, same question
        p1_variants = build_p1_variants(s["image"], s["question"])
        # P3-P7 — same image, 39 text variants
        text_variants = build_text_variants(s["question"])

        if args.model == "llava_med":
            # P1 batch
            p1_items = [(img, q) for _, img, q in p1_variants]
            p1_preds = llava_batch_answer(model, p1_items, batch_size=args.batch_size)
            # Text batch (same image repeated)
            text_items = [(s["image"], full_q) for _, _, _, _, full_q in text_variants]
            text_preds = llava_batch_answer(model, text_items, batch_size=args.batch_size)
        else:
            cands = ["yes","no",str(s["answer"]),"cannot determine","normal","abnormal"]
            cands = list(dict.fromkeys(cands))
            p1_preds = [model.answer(img, q, candidates=cands).get("answer","")
                         for _, img, q in p1_variants]
            text_preds = [model.answer(s["image"], full_q, candidates=cands).get("answer","")
                           for _, _, _, _, full_q in text_variants]

        base_rec = {"sample_id": s["sample_id"], "dataset": args.dataset,
                    "gt": s["answer"], "type": s["type"]}
        # write P1
        for (kind, _, q), pred in zip(p1_variants, p1_preds):
            f_out.write(json.dumps({**base_rec, "probe":"P1","variant":kind,
                                      "question": q, "pred": pred}, default=str) + "\n")
        orig_pred = p1_preds[0]
        # write text variants
        for (probe, vid, pid, ptext, full_q), pred in zip(text_variants, text_preds):
            f_out.write(json.dumps({**base_rec, "probe":probe, "variant":vid,
                                      "prefix_id": pid, "prefix_text": ptext,
                                      "question": full_q, "pred_original": orig_pred,
                                      "pred": pred}, default=str) + "\n")
        f_out.flush()
        if si % 10 == 0:
            print(f"[{si+1}/{len(samples)}] elapsed={time.time()-ts:.1f}s", flush=True)

    f_out.close()
    print(f"done. wrote {out_dir / 'raw.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
