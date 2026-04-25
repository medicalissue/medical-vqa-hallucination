"""Run LLaVA-Med across all (dataset, n) configs in a single process to amortize
the 2-3min model load.
"""
from __future__ import annotations
import sys, time, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datasets_loader import load
from probes import generate_all_variants
from models_wrapper import LlavaMedWrapper

CONFIGS = [
    ("vqa_rad", 30),
    ("vqa_med_2019", 20),
    ("vqa_med_2021", 20),
]

OUT_BASE = Path("/home/ubuntu/mmbert_work/results/llava_med_full")
OUT_BASE.mkdir(parents=True, exist_ok=True)


def main():
    print("[init] loading LLaVA-Med ...", flush=True)
    t0 = time.time()
    model = LlavaMedWrapper()
    print(f"[init] loaded in {time.time()-t0:.1f}s", flush=True)

    for dataset, n in CONFIGS:
        out_dir = OUT_BASE / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl = open(out_dir / "raw.jsonl", "w")
        samples = load(dataset, n)
        print(f"[{dataset}] running {len(samples)} samples", flush=True)
        ts = time.time()
        for si, s in enumerate(samples):
            for probe, v in generate_all_variants(s["image"], s["question"]):
                pred = model.answer(v.image, v.question)
                rec = {
                    "sample_id": s["sample_id"], "type": s["type"],
                    "dataset": dataset, "gt": s["answer"],
                    "probe": probe, "variant": v.variant_id,
                    "question": v.question, "pred": pred.get("answer"),
                    "raw": pred.get("raw"), "meta": v.meta,
                }
                jsonl.write(json.dumps(rec, default=str) + "\n")
                jsonl.flush()
            if si % 5 == 0:
                el = time.time() - ts
                print(f"[{dataset}] [{si+1}/{len(samples)}] elapsed={el:.1f}s", flush=True)
        jsonl.close()
        print(f"[{dataset}] DONE in {time.time()-ts:.1f}s", flush=True)
    print("[all-done]", flush=True)


if __name__ == "__main__":
    main()
