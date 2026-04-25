"""우리 raw.jsonl을 MUMC analysis 형식 CSV로 변환.

MUMC 형식:
  eval_finetuned/{dataset}/results_{dataset}.csv
    qid,question,answer,modality,pred_label,correct
    (= P1 orig variant의 baseline 정확도)

  model_response/{dataset}/p1_image_grounding.csv
    qid,question,modality,gt,pred_original,pred_blank,flip_blank,
    pred_white,flip_white,pred_noise,flip_noise,pred_gray,flip_gray

  model_response/{dataset}/p3_irrelevant_text.csv
    qid,question,modality,gt,prefix_id,prefix_text,pred_original,pred_perturbed,flip

  p4_demographic.csv  (same schema as p3)
  p7_modality_mismatch.csv  (same schema as p3, prefix = false modality statement)

  hallucination_summary.json
    {dataset, n_samples, probes: {p1: {blank, white, noise, gray: {n_total, flip_naive, n_yes_no, flip_yes_no}}, ...}}

본 스크립트:
  - per-model output: results/mumc_export/{model}/{dataset}/...
  - dataset 이름: vqa_med_2019, vqa_med_2021, vqa_rad
"""
from __future__ import annotations
import json, csv, sys, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from semantic_metrics import normalize, contains_answer, extract_yn

OUT = ROOT / "results" / "mumc_export"
OUT.mkdir(exist_ok=True)

MODELS = ["llava_med", "biomed_clip"]
DATASETS = ["vqa_rad", "vqa_med_2019", "vqa_med_2021"]
DS_LABEL = {"vqa_rad": "vqa_rad", "vqa_med_2019": "med2019_local", "vqa_med_2021": "vqa_med2021"}

MODALITY_PATTERNS = [
    # MUMC와 동일한 키워드 (mammography는 MUMC가 unknown 처리하므로 제외)
    ("X-ray",       re.compile(r"\b(x[-\s]?ray|xr\b|chest x|plain film)\b", re.I)),
    ("CT",          re.compile(r"\b(ct\b|cta\b|computed tomograph)", re.I)),
    ("MRI",         re.compile(r"\b(mri\b|mr\b|magnetic resonance)", re.I)),
    ("Ultrasound", re.compile(r"\b(ultrasound|sonograph|doppler|us\b)", re.I)),
    ("Angiography", re.compile(r"\b(angiograph|angiogram)\b", re.I)),
]


def infer_modality(question: str, gt: str = "") -> str:
    """MUMC와 동일하게 질문만 보고 modality 추론 (GT는 무시)."""
    text = question or ""
    for label, pat in MODALITY_PATTERNS:
        if pat.search(text):
            return label
    return "unknown"


def find_raw(model: str, dataset: str) -> Path | None:
    for sub in ("_combined", "_big", "_full"):
        p = ROOT / "results" / f"{model}{sub}" / dataset / "raw.jsonl"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def flip_naive(a: str, b: str) -> int:
    return int(normalize(a) != normalize(b))


def yes_no_subset(records):
    """Filter records where GT is yes/no — closed."""
    return [r for r in records if normalize(r.get("gt", "")) in ("yes", "no")]


def write_eval_finetuned(model, dataset, recs, out_dir: Path):
    """results_{dataset}.csv: qid,question,answer,modality,pred_label,correct
       Uses orig variant from any probe (we use P1_blank orig)."""
    rows = []
    seen = set()
    for r in recs:
        if r["probe"] != "P1_blank" or r["variant"] != "orig": continue
        sid = r["sample_id"]
        if sid in seen: continue
        seen.add(sid)
        modality = infer_modality(r["question"], r.get("gt",""))
        correct = int(contains_answer(r["pred"], r["gt"]))
        rows.append({
            "qid": len(rows),
            "question": r["question"],
            "answer": r.get("gt", ""),
            "modality": modality,
            "pred_label": r["pred"],
            "correct": correct,
        })
    out_path = out_dir / f"results_{DS_LABEL[dataset]}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid","question","answer","modality","pred_label","correct"])
        w.writeheader(); w.writerows(rows)
    print(f"  {out_path.name}: {len(rows)} rows")
    return rows


def write_p1(model, dataset, recs, out_dir: Path):
    """p1_image_grounding.csv: per-sample row with all blank variants."""
    by_sample = defaultdict(dict)
    for r in recs:
        if r["probe"] != "P1_blank": continue
        by_sample[r["sample_id"]][r["variant"]] = r
    rows = []
    for sid, vs in by_sample.items():
        if "orig" not in vs: continue
        orig = vs["orig"]
        modality = infer_modality(orig["question"], orig.get("gt",""))
        row = {
            "qid": len(rows),
            "question": orig["question"],
            "modality": modality,
            "gt": orig.get("gt",""),
            "pred_original": orig["pred"],
        }
        for kind in ("black","white","noise","gray"):
            if kind in vs:
                row[f"pred_{kind}"] = vs[kind]["pred"]
                row[f"flip_{kind}"] = flip_naive(orig["pred"], vs[kind]["pred"])
            else:
                row[f"pred_{kind}"] = ""
                row[f"flip_{kind}"] = ""
        # MUMC uses pred_blank but we keep pred_black to match our naming; map black→blank
        row["pred_blank"] = row.pop("pred_black")
        row["flip_blank"] = row.pop("flip_black")
        rows.append(row)
    out_path = out_dir / "p1_image_grounding.csv"
    fields = ["qid","question","modality","gt","pred_original",
              "pred_blank","flip_blank","pred_white","flip_white",
              "pred_noise","flip_noise","pred_gray","flip_gray"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  {out_path.name}: {len(rows)} rows")
    # summary
    summary = {}
    for kind_out, kind_in in [("blank","black"),("white","white"),("noise","noise"),("gray","gray")]:
        flips = [int(r[f"flip_{kind_out}"]) for r in rows if r[f"flip_{kind_out}"] != ""]
        n_total = len(flips)
        flip_naive_pct = round(100*sum(flips)/n_total, 2) if n_total else 0
        # yes_no subset
        yn_rows = [r for r in rows if normalize(r["gt"]) in ("yes","no")]
        yn_flips = []
        for r in yn_rows:
            ya = extract_yn(r["pred_original"])
            yb = extract_yn(r.get(f"pred_{kind_out}", ""))
            if ya is not None and yb is not None:
                yn_flips.append(int(ya != yb))
        n_yn = len(yn_flips)
        flip_yn = round(100*sum(yn_flips)/n_yn, 2) if n_yn else 0
        summary[kind_out] = {"n_total": n_total, "flip_naive": flip_naive_pct,
                              "n_yes_no": n_yn, "flip_yes_no": flip_yn}
    return summary


def write_perturbation_probe(probe_name, file_name, recs, out_dir: Path):
    """For P3 / P4 probes: per (sample, prefix variant) row."""
    by_sample = defaultdict(dict)
    for r in recs:
        if r["probe"] != probe_name: continue
        by_sample[r["sample_id"]][r["variant"]] = r
    rows = []
    per_prefix_stats = defaultdict(lambda: {"n_total": 0, "flips": 0, "yn_n": 0, "yn_flips": 0,
                                             "prefix": ""})
    for sid, vs in by_sample.items():
        if "orig" not in vs: continue
        orig = vs["orig"]
        modality = infer_modality(orig["question"], orig.get("gt",""))
        for v_id, r in vs.items():
            if v_id == "orig": continue
            prefix_id = v_id  # use the variant_id as prefix_id
            prefix_text = (r.get("meta") or {}).get("prefix") or ""
            if probe_name == "P4_demographic":
                # demographic prefix from meta.prefix
                prefix_text = (r.get("meta") or {}).get("prefix") or ""
            flip = flip_naive(orig["pred"], r["pred"])
            rows.append({
                "qid": len(rows),
                "question": orig["question"],
                "modality": modality,
                "gt": orig.get("gt",""),
                "prefix_id": prefix_id,
                "prefix_text": prefix_text.strip(),
                "pred_original": orig["pred"],
                "pred_perturbed": r["pred"],
                "flip": flip,
            })
            stats = per_prefix_stats[prefix_id]
            stats["prefix"] = prefix_text.strip()
            stats["n_total"] += 1
            stats["flips"] += flip
            ya = extract_yn(orig["pred"]); yb = extract_yn(r["pred"])
            if ya is not None and yb is not None:
                stats["yn_n"] += 1
                stats["yn_flips"] += int(ya != yb)
    out_path = out_dir / file_name
    fields = ["qid","question","modality","gt","prefix_id","prefix_text",
              "pred_original","pred_perturbed","flip"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  {out_path.name}: {len(rows)} rows")
    # summary
    per_prefix = []
    for pid, s in sorted(per_prefix_stats.items()):
        per_prefix.append({
            "prefix_id": pid,
            "prefix": s["prefix"],
            "n_total": s["n_total"],
            "flip_naive": round(100*s["flips"]/s["n_total"], 2) if s["n_total"] else 0,
            "n_yes_no": s["yn_n"],
            "flip_yes_no": round(100*s["yn_flips"]/s["yn_n"], 2) if s["yn_n"] else 0,
        })
    avg = round(sum(p["flip_naive"] for p in per_prefix) / max(1, len(per_prefix)), 2)
    return {"per_prefix": per_prefix, "avg_flip_naive": avg}


def write_p7(model, dataset, p7_recs, out_dir: Path):
    """For P7 modality mismatch: read from p7_{model}_big/ raw.jsonl.
    Schema same as p3."""
    rows = []
    per_prefix_stats = defaultdict(lambda: {"n_total":0,"flips":0,"yn_n":0,"yn_flips":0,"prefix":""})
    # group by sample
    by_sample = defaultdict(list)
    for r in p7_recs: by_sample[r["sample_id"]].append(r)
    for sid, rs in by_sample.items():
        # find 'recog_orig' as the "original"
        orig = next((r for r in rs if r["variant"] == "recog_orig"), None)
        if orig is None: continue
        gt_modality = orig.get("gt_modality", "unknown")
        for r in rs:
            if r["variant"] == "recog_orig": continue
            if r.get("meta", {}).get("subprobe") != "misattribution": continue
            if r["meta"].get("kind") != "wrong": continue
            false_mod = r["meta"].get("false_modality", "")
            prefix_id = f"{r['variant']}"
            prefix_text = f"This image was framed as {false_mod} (true={gt_modality}); Q: {r['question']}"
            flip = flip_naive(orig["pred"], r["pred"])
            rows.append({
                "qid": len(rows),
                "question": r["question"],
                "modality": gt_modality.upper(),
                "gt": orig.get("gt_answer",""),
                "prefix_id": prefix_id,
                "prefix_text": prefix_text,
                "pred_original": orig["pred"],
                "pred_perturbed": r["pred"],
                "flip": flip,
            })
            stats = per_prefix_stats[false_mod]
            stats["prefix"] = f"This image was obtained using {false_mod}."
            stats["n_total"] += 1
            stats["flips"] += flip
    out_path = out_dir / "p7_modality_mismatch.csv"
    fields = ["qid","question","modality","gt","prefix_id","prefix_text",
              "pred_original","pred_perturbed","flip"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  {out_path.name}: {len(rows)} rows")
    per_prefix = []
    for i, (fm, s) in enumerate(sorted(per_prefix_stats.items()), start=1):
        per_prefix.append({
            "prefix_id": i,
            "prefix": s["prefix"],
            "n_total": s["n_total"],
            "flip_naive": round(100*s["flips"]/s["n_total"], 2) if s["n_total"] else 0,
        })
    avg = round(sum(p["flip_naive"] for p in per_prefix) / max(1,len(per_prefix)), 2)
    return {"per_prefix": per_prefix, "avg_flip_naive": avg}


def main():
    for model in MODELS:
        print(f"\n=== {model} ===")
        for dataset in DATASETS:
            p = find_raw(model, dataset)
            if not p:
                print(f"[{dataset}] no raw.jsonl, skipping")
                continue
            recs = [json.loads(l) for l in open(p)]
            ds_label = DS_LABEL[dataset]
            ef_dir = OUT / model / "eval_finetuned" / ds_label
            mr_dir = OUT / model / "model_response" / ds_label
            ef_dir.mkdir(parents=True, exist_ok=True)
            mr_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{dataset}] {len(recs)} records")

            # 1) eval_finetuned/results_*.csv
            eval_rows = write_eval_finetuned(model, dataset, recs, ef_dir)

            # 2) p1
            p1_summary = write_p1(model, dataset, recs, mr_dir)

            # 3) p3
            p3_summary = write_perturbation_probe("P3_prefix", "p3_irrelevant_text.csv", recs, mr_dir)

            # 4) p4 (demographic — MUMC's p4 uses one prefix per row; ours has multiple demos)
            p4_summary = write_perturbation_probe("P4_demographic", "p4_demographic.csv", recs, mr_dir)

            # 5) P7 mismatch — only present for some datasets
            p7_recs = []
            for sub in ("_big",""):
                p7p = ROOT / "results" / f"p7_{model}{sub}" / "raw.jsonl"
                if p7p.exists() and p7p.stat().st_size > 0:
                    all_p7 = [json.loads(l) for l in open(p7p)]
                    p7_recs = [r for r in all_p7 if r.get("dataset") == dataset]
                    if p7_recs: break
            if p7_recs:
                p7_summary = write_p7(model, dataset, p7_recs, mr_dir)
            else:
                p7_summary = None

            # summary.json
            summary = {
                "dataset": ds_label,
                "model": model,
                "n_samples": len({r["sample_id"] for r in recs}),
                "probes": {"p1": p1_summary, "p3": p3_summary, "p4": p4_summary},
            }
            if p7_summary:
                summary["probes"]["p7"] = p7_summary
            with open(mr_dir / "hallucination_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  hallucination_summary.json")

    print(f"\nALL DONE → {OUT}")


if __name__ == "__main__":
    main()
