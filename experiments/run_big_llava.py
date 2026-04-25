"""LLaVA-Med large-scale sweep — different seeds to add samples on top of run_all_llava."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datasets_loader import load
from probes import generate_all_variants
from models_wrapper import LlavaMedWrapper

CONFIGS = [
    ("vqa_rad",      80),
    ("vqa_med_2019", 60),
    ("vqa_med_2021", 60),
]
SEED = 42  # different seed → disjoint samples from run_all_llava (seed=0)

OUT_BASE = Path("/home/ubuntu/mmbert_work/results/llava_med_big")
OUT_BASE.mkdir(parents=True, exist_ok=True)


def main():
    print("[init] loading LLaVA-Med ...", flush=True)
    t0 = time.time()
    model = LlavaMedWrapper()
    print(f"[init] loaded in {time.time()-t0:.1f}s", flush=True)

    for dataset, n in CONFIGS:
        out = OUT_BASE / dataset; out.mkdir(parents=True, exist_ok=True)
        jsonl = open(out / "raw.jsonl", "w")
        samples = load(dataset, n, seed=SEED)
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
                jsonl.write(json.dumps(rec, default=str) + "\n"); jsonl.flush()
            if si % 5 == 0:
                print(f"[{dataset}] [{si+1}/{len(samples)}] elapsed={time.time()-ts:.1f}s", flush=True)
        jsonl.close()
        print(f"[{dataset}] DONE in {time.time()-ts:.1f}s", flush=True)
    print("[all-done]", flush=True)


if __name__ == "__main__":
    main()
