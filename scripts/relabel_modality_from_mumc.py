"""우리 CSV의 modality 라벨을 MUMC가 사용한 것과 동일하게 맞춤.

전략:
1. MUMC의 모든 CSV에서 (question → modality) 매핑 dict 구축
2. 우리 CSV의 각 row에 대해, 같은 question이 매핑에 있으면 그 modality 사용
3. 없으면 fallback으로 기존 키워드 휴리스틱

VQA-RAD에서 MUMC가 \"chest/lung/rib\" 키워드 질문을 X-ray로 라벨링한 것 등 자동 흡수.
"""
from __future__ import annotations
import csv, json, sys, re
from pathlib import Path

REPO = Path(__file__).parent.parent
TARGET = Path("/Users/medicalissue/Desktop/medical")
sys.path.insert(0, str(REPO / "experiments"))
from semantic_metrics import normalize, extract_yn

MUMC_BASE = TARGET / "analysis_mumc"
MODELS = ["llava_med", "biomed_clip"]
DS = ["vqa_rad", "med2019_local", "vqa_med2021"]
MUMC_DS = {"vqa_rad": "vqa_rad", "med2019_local": "vqa_med_2019", "vqa_med2021": "vqa_med_2021"}

# Fallback 키워드 (MUMC 매핑에 없는 새 question에 사용)
FALLBACK_PATTERNS = [
    ("X-ray",       re.compile(r"\b(x[-\s]?ray|xr\b|chest x|plain film)\b", re.I)),
    ("CT",          re.compile(r"\b(ct\b|cta\b|computed tomograph)", re.I)),
    ("MRI",         re.compile(r"\b(mri\b|mr\b|magnetic resonance)", re.I)),
    ("Ultrasound",  re.compile(r"\b(ultrasound|sonograph|doppler|us\b)", re.I)),
    ("Angiography", re.compile(r"\b(angiograph|angiogram)\b", re.I)),
]


def normalize_question(q: str) -> str:
    """비교용으로 question 정규화 (대소문자, 공백, 끝 부호)."""
    return " ".join(str(q).lower().strip().split()).rstrip(".!?")


def build_mumc_lookup() -> dict:
    """모든 MUMC CSV에서 (dataset, normalized_question) → modality."""
    lookup = {}
    for ds_short, ds_full in MUMC_DS.items():
        # MUMC 폴더 구조: model_response/{ds_full}/p{N}_*.csv
        mumc_dir = MUMC_BASE / "model_response" / ds_full
        if not mumc_dir.exists(): continue
        for csv_path in mumc_dir.glob("p*.csv"):
            for row in csv.DictReader(open(csv_path)):
                q = normalize_question(row.get("question", ""))
                m = row.get("modality", "unknown")
                if not q: continue
                # 첫 등장만 사용 (multiple votes 무시)
                if (ds_short, q) not in lookup:
                    lookup[(ds_short, q)] = m
        # 또한 eval CSV도
        eval_csv = MUMC_BASE / "eval_finetuned" / ds_full / f"results_{ds_full}.csv"
        if eval_csv.exists():
            for row in csv.DictReader(open(eval_csv)):
                q = normalize_question(row.get("question", ""))
                m = row.get("modality", "unknown")
                if q and (ds_short, q) not in lookup:
                    lookup[(ds_short, q)] = m
    return lookup


def fallback_modality(q: str) -> str:
    """질문 텍스트에서 modality 키워드 검색 (P3-P7 prefix는 strip)."""
    text = q or ""
    for label, pat in FALLBACK_PATTERNS:
        if pat.search(text): return label
    return "unknown"


def get_orig_question(q: str, prefix_text: str = "") -> str:
    """prefix가 있으면 떼고 원래 질문만 반환."""
    if prefix_text and q.startswith(prefix_text):
        return q[len(prefix_text):].strip()
    # 일반적으로 prefix가 \". \" 로 끝나는 sentence
    if ". " in q:
        # 첫 \". \" 이후가 질문일 가능성
        parts = q.split(". ", 1)
        if len(parts) == 2 and len(parts[1]) > 10:
            return parts[1].strip()
    return q.strip()


def relabel_csv(csv_path: Path, ds_short: str, lookup: dict, has_prefix: bool):
    """In-place relabel modality column."""
    rows = list(csv.DictReader(open(csv_path)))
    if not rows: return 0, 0
    field_names = list(rows[0].keys())
    matched = 0
    fallback = 0
    for r in rows:
        q = r.get("question", "")
        # prefix 있으면 원래 질문만 추출 (P3-P7)
        if has_prefix:
            pref = r.get("prefix_text", "")
            orig_q = get_orig_question(q, pref)
        else:
            orig_q = q
        norm = normalize_question(orig_q)
        new_mod = lookup.get((ds_short, norm))
        if new_mod is not None:
            matched += 1
            r["modality"] = new_mod
        else:
            fallback += 1
            r["modality"] = fallback_modality(orig_q)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_names)
        w.writeheader()
        w.writerows(rows)
    return matched, fallback


def main():
    lookup = build_mumc_lookup()
    print(f"MUMC lookup: {len(lookup)} (dataset, question) → modality entries\n")

    for model in MODELS:
        folder_root = TARGET / f"analysis_{model}"
        for ds in DS:
            for sub in ("vqa_accuracy", "hallucination_probes"):
                ds_dir = folder_root / sub / ds
                if not ds_dir.exists(): continue
                for csv_path in sorted(ds_dir.glob("*.csv")):
                    has_prefix = csv_path.stem.startswith("p") and csv_path.stem != "p1_image_grounding"
                    m, f = relabel_csv(csv_path, ds, lookup, has_prefix)
                    print(f"  {model}/{sub}/{ds}/{csv_path.name}: matched={m} fallback={f}")

    # mumc_export 도 동일하게 동기화
    for model in MODELS:
        for sub in ("eval_finetuned", "model_response"):
            base = REPO / "results" / "mumc_export" / model / sub
            if not base.exists(): continue
            for ds_dir in base.iterdir():
                if not ds_dir.is_dir(): continue
                ds = ds_dir.name  # vqa_rad / med2019_local / vqa_med2021
                for csv_path in sorted(ds_dir.glob("*.csv")):
                    has_prefix = csv_path.stem.startswith("p") and csv_path.stem != "p1_image_grounding"
                    m, f = relabel_csv(csv_path, ds, lookup, has_prefix)
                    if m + f > 0:
                        print(f"  mumc_export/{model}/{sub}/{ds}/{csv_path.name}: matched={m} fallback={f}")


if __name__ == "__main__":
    main()
