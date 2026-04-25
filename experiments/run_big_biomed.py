"""BiomedCLIP large-scale sweep — replaces previous n=30 results."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datasets_loader import load
from probes import generate_all_variants
from models_wrapper import BiomedClipWrapper

CONFIGS = [
    ("vqa_rad",      150),
    ("vqa_med_2019", 150),
    ("vqa_med_2021", 150),
]

OUT_BASE = Path("/home/ubuntu/mmbert_work/results/biomed_clip_big")
OUT_BASE.mkdir(parents=True, exist_ok=True)


def main():
    print("[init] loading BiomedCLIP ...", flush=True)
    model = BiomedClipWrapper()
    for dataset, n in CONFIGS:
        out = OUT_BASE / dataset; out.mkdir(parents=True, exist_ok=True)
        jsonl = open(out / "raw.jsonl", "w")
        samples = load(dataset, n)
        print(f"[{dataset}] running {len(samples)} samples", flush=True)
        ts = time.time()
        for si, s in enumerate(samples):
            cands = ["yes", "no", str(s["answer"]), "cannot determine", "normal", "abnormal"]
            cands = list(dict.fromkeys(cands))
            for probe, v in generate_all_variants(s["image"], s["question"]):
                pred = model.answer(v.image, v.question, candidates=cands)
                rec = {
                    "sample_id": s["sample_id"], "type": s["type"],
                    "dataset": dataset, "gt": s["answer"],
                    "probe": probe, "variant": v.variant_id,
                    "question": v.question, "pred": pred.get("answer"),
                    "raw": pred.get("raw"), "confidence": pred.get("confidence"),
                    "candidates": cands, "meta": v.meta,
                }
                jsonl.write(json.dumps(rec, default=str) + "\n"); jsonl.flush()
            if si % 25 == 0:
                print(f"[{dataset}] [{si+1}/{len(samples)}] elapsed={time.time()-ts:.1f}s", flush=True)
        jsonl.close()
        print(f"[{dataset}] DONE in {time.time()-ts:.1f}s", flush=True)


if __name__ == "__main__":
    main()
