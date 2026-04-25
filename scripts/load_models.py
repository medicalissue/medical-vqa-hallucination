"""Sanity-check: load both models and run 1 inference each on a VQA-RAD sample.

Models:
  - chaoyinshe/llava-med-v1.5-mistral-7b-hf  (generative; fp16)
  - xmcmic/MedVInT-TE                         (classification; full precision)

The first time this runs, ~20-30 GB of weights are pulled into HF_HOME.
"""
from __future__ import annotations
import os, sys, time, json, argparse
import torch
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
DATA = Path("/home/ubuntu/mmbert_work/data")


def load_one_sample():
    from datasets import load_from_disk
    ds = load_from_disk(str(DATA / "vqa_rad" / "hf"))
    ex = ds["test"][0]
    return ex["image"], ex["question"], ex["answer"]


def test_llava_med():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    print(f"[llava-med] loading {model_id} ...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda:0"
    )
    print(f"[llava-med] loaded in {time.time()-t0:.1f}s", flush=True)

    image, question, gt = load_one_sample()
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0", torch.float16)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    print(f"[llava-med] Q: {question}\n[llava-med] GT: {gt}\n[llava-med] PRED: {decoded!r}")
    del model; torch.cuda.empty_cache()


def test_medvint():
    # MedVInT uses a custom repo; try loading via trust_remote_code
    from transformers import AutoModel, AutoTokenizer
    model_id = "xmcmic/MedVInT-TE"
    print(f"[medvint] loading {model_id} ...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).cuda()
        print(f"[medvint] loaded. param count: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    except Exception as e:
        print(f"[medvint] failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["llava", "medvint", "both"], default="both")
    args = ap.parse_args()
    if args.only in ("llava", "both"): test_llava_med()
    if args.only in ("medvint", "both"): test_medvint()
