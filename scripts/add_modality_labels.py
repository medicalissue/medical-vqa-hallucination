"""각 sample에 modality 라벨 추가.

각 raw.jsonl 옆에 raw_labeled.jsonl 생성. 추가되는 필드:
  - category: VQA-Med 2019의 4 카테고리 (modality/plane/organ/abnormality), 다른 dataset에서는 추론
  - modality: x-ray | ct | mri | ultrasound | angiography | other | unknown
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent

# Modality 추론 키워드 (질문 + GT 둘 다 확인)
MODALITY_PATTERNS = [
    ("x-ray",        re.compile(r"\b(x[-\s]?ray|xr|chest x|plain film|radiograph)\b", re.I)),
    ("ct",           re.compile(r"\b(ct|cta|computed tomograph|axial ct)\b", re.I)),
    ("mri",          re.compile(r"\b(mri|mr|magnetic resonance|t1|t2|flair|dwi|adc)\b", re.I)),
    ("ultrasound",   re.compile(r"\b(us|ultrasound|sonograph|doppler|echocardio|usd)\b", re.I)),
    ("angiography",  re.compile(r"\b(angiograph|angiogram)\b", re.I)),
    ("nuclear",      re.compile(r"\b(pet|spect|scintigraph|nuclear)\b", re.I)),
    ("mammography",  re.compile(r"\bmammograph\b", re.I)),
    ("endoscopy",    re.compile(r"\bendoscop\b", re.I)),
]


def infer_modality(text: str) -> str:
    """text(질문+GT 합친 것)에서 modality 추론. 가장 먼저 매치되는 것."""
    if not text: return "unknown"
    for label, pat in MODALITY_PATTERNS:
        if pat.search(text):
            return label
    return "unknown"


def category_from_sample_id(sample_id: str) -> str:
    """VQA-Med 2019 sample_id에서 카테고리 추출."""
    if "_modality" in sample_id: return "modality"
    if "_plane" in sample_id: return "plane"
    if "_organ" in sample_id: return "organ"
    if "_abnormality" in sample_id: return "abnormality"
    return ""


def label_one_record(r: dict) -> dict:
    cat = category_from_sample_id(r["sample_id"])
    if cat:
        r["category"] = cat
    # 모달리티는 (GT + 질문)을 함께 검사
    text = f"{r.get('gt','')} {r.get('question','')}"
    modality = infer_modality(text)
    # VQA-Med 2019 modality 카테고리 답 자체에 modality 단어 그대로 들어있음 → 강한 신호
    if cat == "modality":
        # GT를 직접 modality로 매핑
        gt_mod = infer_modality(str(r.get("gt", "")))
        if gt_mod != "unknown":
            modality = gt_mod
    r["modality"] = modality
    return r


def process_file(in_path: Path, out_path: Path):
    counts = Counter()
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            r = json.loads(line)
            r = label_one_record(r)
            fout.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            counts[r["modality"]] += 1
    return counts


def main():
    targets = []
    for sub in ("biomed_clip_big", "biomed_clip_full",
                "llava_med_combined", "llava_med_big", "llava_med_full"):
        for d in ("vqa_rad", "vqa_med_2019", "vqa_med_2021"):
            p = ROOT / "results" / sub / d / "raw.jsonl"
            if p.exists() and p.stat().st_size > 0:
                targets.append((sub, d, p))

    summary = {}
    for sub, d, p in targets:
        out = p.parent / "raw_labeled.jsonl"
        counts = process_file(p, out)
        summary[(sub, d)] = counts
        print(f"{sub}/{d}: {sum(counts.values())} records → {out.name}")
        print(f"   modality dist: {dict(counts)}")

    # write summary CSV
    import csv
    summary_path = ROOT / "results" / "modality_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        all_modalities = sorted({m for c in summary.values() for m in c})
        w.writerow(["model_run", "dataset", "total"] + all_modalities)
        for (sub, d), c in summary.items():
            row = [sub, d, sum(c.values())] + [c.get(m, 0) for m in all_modalities]
            w.writerow(row)
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
