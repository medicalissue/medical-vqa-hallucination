"""Merge llava_med_full (seed=0) and llava_med_big (seed=42) into a single combined raw.jsonl per dataset.

The two runs use disjoint random seeds so samples are largely non-overlapping.
We dedupe on sample_id + variant just in case.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = [ROOT / "results" / "llava_med_full", ROOT / "results" / "llava_med_big"]
OUT = ROOT / "results" / "llava_med_combined"
OUT.mkdir(exist_ok=True)

DATASETS = ["vqa_rad", "vqa_med_2019", "vqa_med_2021"]

for d in DATASETS:
    seen = set()
    out_path = OUT / d
    out_path.mkdir(exist_ok=True)
    out_f = open(out_path / "raw.jsonl", "w")
    n = 0
    for src in SRC:
        p = src / d / "raw.jsonl"
        if not p.exists(): continue
        for line in open(p):
            r = json.loads(line)
            key = (r["sample_id"], r["probe"], r["variant"])
            if key in seen: continue
            seen.add(key)
            out_f.write(line)
            n += 1
    out_f.close()
    n_samples = len({k[0] for k in seen})
    print(f"{d}: {n} records / {n_samples} samples")
