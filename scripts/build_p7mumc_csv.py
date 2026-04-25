"""Convert p7mumc raw.jsonl outputs into MUMC-format p7_modality_mismatch.csv
and update each model's analysis_*/model_response/*/p7_modality_mismatch.csv +
hallucination_summary.json.
"""
from __future__ import annotations
import json, csv, sys, re
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).parent.parent
TARGET = Path("/Users/medicalissue/Desktop/medical")
sys.path.insert(0, str(REPO / "experiments"))
from semantic_metrics import normalize, extract_yn

DS_SHORT = {"vqa_rad": "vqa_rad", "vqa_med_2019": "med2019_local", "vqa_med_2021": "vqa_med2021"}
MODEL_FOLDER = {"llava_med": "analysis_llava_med", "biomed_clip": "analysis_biomed_clip"}

MODALITY_PATTERNS = [
    # MUMC와 동일한 키워드 set (mammography 제외, MUMC는 unknown 처리)
    ("X-ray",       re.compile(r"\b(x[-\s]?ray|xr\b|chest x|plain film)\b", re.I)),
    ("CT",          re.compile(r"\b(ct\b|cta\b|computed tomograph)", re.I)),
    ("MRI",         re.compile(r"\b(mri\b|mr\b|magnetic resonance)", re.I)),
    ("Ultrasound", re.compile(r"\b(ultrasound|sonograph|doppler|us\b)", re.I)),
    ("Angiography", re.compile(r"\b(angiograph|angiogram)\b", re.I)),
]

def infer_modality(question: str, gt: str = "") -> str:
    """MUMC와 동일: 질문 텍스트만 보고 modality 추론. GT는 사용 안 함."""
    text = question or ""
    for label, pat in MODALITY_PATTERNS:
        if pat.search(text): return label
    return "unknown"


def process(model: str, ds: str):
    src = REPO / "results" / f"p7mumc_{model}" / ds / "raw.jsonl"
    if not src.exists():
        print(f"[skip] {src} missing")
        return None
    recs = [json.loads(l) for l in open(src)]
    by_sample = defaultdict(dict)
    for r in recs:
        by_sample[r["sample_id"]][r["variant"]] = r

    rows = []
    per_prefix = defaultdict(lambda: {"flips": 0, "n_total": 0,
                                       "yn_flips": 0, "yn_n": 0, "prefix": ""})
    for sid, vs in by_sample.items():
        if "orig" not in vs: continue
        orig = vs["orig"]
        modality = infer_modality(orig["question"], orig.get("gt", ""))
        for vid, r in vs.items():
            if vid == "orig": continue
            pid = r["prefix_id"]
            ptext = r["prefix_text"]
            flip = int(normalize(orig["pred"]) != normalize(r["pred"]))
            rows.append({
                "qid": len(rows),
                "question": orig["question"],
                "modality": modality,
                "gt": orig.get("gt", ""),
                "prefix_id": pid,
                "prefix_text": ptext,
                "pred_original": orig["pred"],
                "pred_perturbed": r["pred"],
                "flip": flip,
            })
            stats = per_prefix[pid]
            stats["prefix"] = ptext
            stats["n_total"] += 1
            stats["flips"] += flip
            ya, yb = extract_yn(orig["pred"]), extract_yn(r["pred"])
            if ya is not None and yb is not None:
                stats["yn_n"] += 1
                stats["yn_flips"] += int(ya != yb)

    # Write CSV into both target locations: mumc_export and analysis_<model>
    ds_short = DS_SHORT[ds]
    for base in [REPO / "results/mumc_export" / model,
                 TARGET / MODEL_FOLDER[model]]:
        out = base / "model_response" / ds_short / "p7_modality_mismatch.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["qid","question","modality","gt",
                                                "prefix_id","prefix_text",
                                                "pred_original","pred_perturbed","flip"])
            w.writeheader()
            w.writerows(rows)

    # Build summary
    per_prefix_list = []
    for pid in sorted(per_prefix.keys(), key=int):
        s = per_prefix[pid]
        per_prefix_list.append({
            "prefix_id": int(pid),
            "prefix": s["prefix"],
            "n_total": s["n_total"],
            "flip_naive": round(100*s["flips"]/s["n_total"], 2) if s["n_total"] else 0,
            "n_yes_no": s["yn_n"],
            "flip_yes_no": round(100*s["yn_flips"]/s["yn_n"], 2) if s["yn_n"] else 0,
        })
    avg = round(sum(p["flip_naive"] for p in per_prefix_list) / max(1, len(per_prefix_list)), 2)
    return {"per_prefix": per_prefix_list, "avg_flip_naive": avg, "n_samples": len(by_sample), "ds_short": ds_short}


def update_summary(model: str, ds: str, p7_summary):
    if p7_summary is None: return
    ds_short = p7_summary["ds_short"]
    for base in [REPO / "results/mumc_export" / model,
                 TARGET / MODEL_FOLDER[model]]:
        sp = base / "model_response" / ds_short / "hallucination_summary.json"
        if not sp.exists(): continue
        d = json.load(open(sp))
        d["probes"]["p7"] = {
            "per_prefix": p7_summary["per_prefix"],
            "avg_flip_naive": p7_summary["avg_flip_naive"],
        }
        json.dump(d, open(sp, "w"), indent=2)
    print(f"[{model}/{ds}] p7 avg={p7_summary['avg_flip_naive']:.1f}%  n={p7_summary['n_samples']}")
    for p in p7_summary["per_prefix"]:
        print(f"   {p['prefix']:<45} flip_naive={p['flip_naive']:.1f}%  flip_yn={p['flip_yes_no']:.1f}% (n_yn={p['n_yes_no']})")


def main():
    for model in ("biomed_clip", "llava_med"):
        for ds in ("vqa_rad", "vqa_med_2019", "vqa_med_2021"):
            s = process(model, ds)
            update_summary(model, ds, s)


if __name__ == "__main__":
    main()
