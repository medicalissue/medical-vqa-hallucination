"""Convert P5/P6 raw.jsonl outputs into MUMC-format CSV+summary update."""
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
    ("X-ray",       re.compile(r"\b(x[-\s]?ray|xr\b|chest x|plain film)\b", re.I)),
    ("CT",          re.compile(r"\b(ct\b|cta\b|computed tomograph)", re.I)),
    ("MRI",         re.compile(r"\b(mri\b|mr\b|magnetic resonance)", re.I)),
    ("Ultrasound",  re.compile(r"\b(ultrasound|sonograph|doppler|us\b)", re.I)),
    ("Angiography", re.compile(r"\b(angiograph|angiogram)\b", re.I)),
]


def infer_modality(question: str) -> str:
    for label, pat in MODALITY_PATTERNS:
        if pat.search(question or ""): return label
    return "unknown"


def process(probe: str, model: str, ds: str, csv_filename: str):
    src = REPO / "results" / f"{probe}mumc_{model}" / ds / "raw.jsonl"
    if not src.exists(): return None
    recs = [json.loads(l) for l in open(src)]
    by_sample = defaultdict(dict)
    for r in recs:
        by_sample[r["sample_id"]][r["variant"]] = r

    rows = []
    per_prefix = defaultdict(lambda: {"flips":0,"n_total":0,"yn_flips":0,"yn_n":0,"prefix":""})
    for sid, vs in by_sample.items():
        if "orig" not in vs: continue
        orig = vs["orig"]
        modality = infer_modality(orig["question"])
        for vid, r in vs.items():
            if vid == "orig": continue
            pid = r["prefix_id"]
            ptext = r["prefix_text"]
            flip = int(normalize(orig["pred"]) != normalize(r["pred"]))
            rows.append({
                "qid": len(rows),
                "question": orig["question"],
                "modality": modality,
                "gt": orig.get("gt",""),
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

    ds_short = DS_SHORT[ds]
    for base, sub in [(REPO / "results/mumc_export" / model, "model_response"),
                       (TARGET / MODEL_FOLDER[model], "hallucination_probes")]:
        out = base / sub / ds_short / csv_filename
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["qid","question","modality","gt",
                                                "prefix_id","prefix_text",
                                                "pred_original","pred_perturbed","flip"])
            w.writeheader()
            w.writerows(rows)

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
    avg = round(sum(p["flip_naive"] for p in per_prefix_list) / max(1,len(per_prefix_list)), 2)
    return {"per_prefix": per_prefix_list, "avg_flip_naive": avg, "ds_short": ds_short}


def update_summary(probe, model, ds, summary):
    if summary is None: return
    ds_short = summary["ds_short"]
    for base, sub in [(REPO / "results/mumc_export" / model, "model_response"),
                       (TARGET / MODEL_FOLDER[model], "hallucination_probes")]:
        sp = base / sub / ds_short / "hallucination_summary.json"
        if not sp.exists(): continue
        d = json.load(open(sp))
        d["probes"][probe] = {
            "per_prefix": summary["per_prefix"],
            "avg_flip_naive": summary["avg_flip_naive"],
        }
        json.dump(d, open(sp, "w"), indent=2)
    print(f"[{model}/{ds}] {probe} avg={summary['avg_flip_naive']:.1f}%")


def main():
    files = {"p5": "p5_medical_history.csv", "p6": "p6_socioeconomic.csv"}
    for model in ("biomed_clip", "llava_med"):
        for ds in ("vqa_rad", "vqa_med_2019", "vqa_med_2021"):
            for probe, fname in files.items():
                s = process(probe, model, ds, fname)
                update_summary(probe, model, ds, s)


if __name__ == "__main__":
    main()
