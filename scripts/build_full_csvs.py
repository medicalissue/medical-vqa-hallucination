"""Build MUMC-format CSVs from full_{model}/{dataset}/raw.jsonl.

Each raw.jsonl has all 6 probes interleaved. We split per-probe and write:
  vqa_accuracy/{ds_short}/results_{ds_short}.csv      (orig variants)
  hallucination_probes/{ds_short}/p1_image_grounding.csv
  hallucination_probes/{ds_short}/p3_irrelevant_text.csv
  hallucination_probes/{ds_short}/p4_demographic.csv
  hallucination_probes/{ds_short}/p5_medical_history.csv
  hallucination_probes/{ds_short}/p6_socioeconomic.csv
  hallucination_probes/{ds_short}/p7_modality_mismatch.csv
  hallucination_probes/{ds_short}/hallucination_summary.json

Modality 라벨: MUMC lookup 사용.
"""
from __future__ import annotations
import json, csv, sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).parent.parent
TARGET = Path("/Users/medicalissue/Desktop/medical")
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "scripts"))
from semantic_metrics import normalize, contains_answer, extract_yn
from relabel_modality_from_mumc import build_mumc_lookup, normalize_question, fallback_modality

DS_SHORT = {"vqa_rad": "vqa_rad", "vqa_med_2019": "med2019_local", "vqa_med_2021": "vqa_med2021"}
MODEL_FOLDER = {"llava_med": "analysis_llava_med", "biomed_clip": "analysis_biomed_clip"}
PROBE_FILE = {
    "P3": "p3_irrelevant_text.csv",
    "P4": "p4_demographic.csv",
    "P5": "p5_medical_history.csv",
    "P6": "p6_socioeconomic.csv",
    "P7": "p7_modality_mismatch.csv",
}


def get_modality(ds_short: str, question: str, prefix_text: str, lookup: dict) -> str:
    """prefix 떼고 원본 질문으로 MUMC lookup."""
    q = question
    if prefix_text and q.startswith(prefix_text):
        q = q[len(prefix_text):].strip()
    norm = normalize_question(q)
    return lookup.get((ds_short, norm)) or fallback_modality(q)


def process(model: str, dataset: str, lookup: dict):
    src = REPO / "results" / f"full_{model}" / dataset / "raw.jsonl"
    if not src.exists():
        print(f"[skip] {src}")
        return
    recs = [json.loads(l) for l in open(src)]
    ds_short = DS_SHORT[dataset]
    target_root = TARGET / MODEL_FOLDER[model]
    va_dir = target_root / "vqa_accuracy" / ds_short
    hp_dir = target_root / "hallucination_probes" / ds_short
    va_dir.mkdir(parents=True, exist_ok=True)
    hp_dir.mkdir(parents=True, exist_ok=True)

    # also push to mumc_export
    me_va = REPO / "results/mumc_export" / model / "eval_finetuned" / ds_short
    me_hp = REPO / "results/mumc_export" / model / "model_response" / ds_short
    me_va.mkdir(parents=True, exist_ok=True)
    me_hp.mkdir(parents=True, exist_ok=True)

    # group by sample
    by_sample = defaultdict(dict)
    for r in recs:
        by_sample[r["sample_id"]][(r["probe"], r["variant"])] = r

    # ---------- 1) vqa_accuracy/results_{ds_short}.csv ----------
    eval_rows = []
    for sid, rs in by_sample.items():
        orig = rs.get(("P1", "orig"))
        if orig is None: continue
        modality = get_modality(ds_short, orig["question"], "", lookup)
        eval_rows.append({
            "qid": len(eval_rows),
            "question": orig["question"],
            "answer": orig.get("gt", ""),
            "modality": modality,
            "pred_label": orig["pred"],
            "correct": int(contains_answer(orig["pred"], orig.get("gt",""))),
        })
    for d in (va_dir, me_va):
        with open(d / f"results_{ds_short}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["qid","question","answer","modality","pred_label","correct"])
            w.writeheader(); w.writerows(eval_rows)

    # ---------- 2) p1_image_grounding.csv ----------
    p1_rows = []
    p1_summary = {}
    for sid, rs in by_sample.items():
        orig_r = rs.get(("P1", "orig"))
        if orig_r is None: continue
        row = {
            "qid": len(p1_rows),
            "question": orig_r["question"],
            "modality": get_modality(ds_short, orig_r["question"], "", lookup),
            "gt": orig_r.get("gt",""),
            "pred_original": orig_r["pred"],
        }
        for kind in ("blank","white","noise","gray"):
            v = rs.get(("P1", kind))
            if v is not None:
                row[f"pred_{kind}"] = v["pred"]
                row[f"flip_{kind}"] = int(normalize(orig_r["pred"]) != normalize(v["pred"]))
            else:
                row[f"pred_{kind}"] = ""; row[f"flip_{kind}"] = ""
        p1_rows.append(row)
    fields_p1 = ["qid","question","modality","gt","pred_original",
                  "pred_blank","flip_blank","pred_white","flip_white",
                  "pred_noise","flip_noise","pred_gray","flip_gray"]
    for d in (hp_dir, me_hp):
        with open(d / "p1_image_grounding.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields_p1)
            w.writeheader(); w.writerows(p1_rows)
    # P1 summary
    for kind in ("blank","white","noise","gray"):
        flips = [int(r[f"flip_{kind}"]) for r in p1_rows if r.get(f"flip_{kind}") not in ("", None)]
        n_total = len(flips)
        flip_n = round(100*sum(flips)/n_total, 2) if n_total else 0
        # yes/no subset
        yn = []
        for r in p1_rows:
            ya, yb = extract_yn(r["pred_original"]), extract_yn(r.get(f"pred_{kind}",""))
            if ya is not None and yb is not None: yn.append(int(ya != yb))
        flip_yn = round(100*sum(yn)/len(yn), 2) if yn else 0
        p1_summary[kind] = {"n_total": n_total, "flip_naive": flip_n,
                              "n_yes_no": len(yn), "flip_yes_no": flip_yn}

    # ---------- 3) P3-P7 ----------
    summary = {"dataset": ds_short, "model": model, "n_samples": len(by_sample),
               "probes": {"p1": p1_summary}}

    for probe in ("P3","P4","P5","P6","P7"):
        rows = []
        per_prefix = defaultdict(lambda: {"n_total":0,"flips":0,"yn_n":0,"yn_flips":0,"prefix":""})
        for sid, rs in by_sample.items():
            orig_r = rs.get(("P1","orig"))
            if orig_r is None: continue
            modality = get_modality(ds_short, orig_r["question"], "", lookup)
            for (p, vid), r in rs.items():
                if p != probe: continue
                pid = r.get("prefix_id", vid)
                ptext = r.get("prefix_text", "")
                flip = int(normalize(orig_r["pred"]) != normalize(r["pred"]))
                rows.append({
                    "qid": len(rows),
                    "question": orig_r["question"],
                    "modality": modality,
                    "gt": orig_r.get("gt",""),
                    "prefix_id": pid,
                    "prefix_text": ptext,
                    "pred_original": orig_r["pred"],
                    "pred_perturbed": r["pred"],
                    "flip": flip,
                })
                stats = per_prefix[pid]
                stats["prefix"] = ptext
                stats["n_total"] += 1; stats["flips"] += flip
                ya, yb = extract_yn(orig_r["pred"]), extract_yn(r["pred"])
                if ya is not None and yb is not None:
                    stats["yn_n"] += 1; stats["yn_flips"] += int(ya != yb)
        fname = PROBE_FILE[probe]
        for d in (hp_dir, me_hp):
            with open(d / fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["qid","question","modality","gt",
                                                    "prefix_id","prefix_text",
                                                    "pred_original","pred_perturbed","flip"])
                w.writeheader(); w.writerows(rows)
        # summary
        per_prefix_list = []
        # sort prefix_ids: P4 has tag strings, others int
        try:
            ordered = sorted(per_prefix.keys(), key=lambda x: int(x))
        except (ValueError, TypeError):
            ordered = sorted(per_prefix.keys(), key=str)
        for pid in ordered:
            s = per_prefix[pid]
            per_prefix_list.append({
                "prefix_id": pid,
                "prefix": s["prefix"],
                "n_total": s["n_total"],
                "flip_naive": round(100*s["flips"]/s["n_total"], 2) if s["n_total"] else 0,
                "n_yes_no": s["yn_n"],
                "flip_yes_no": round(100*s["yn_flips"]/s["yn_n"], 2) if s["yn_n"] else 0,
            })
        avg = round(sum(p["flip_naive"] for p in per_prefix_list)/max(1,len(per_prefix_list)), 2)
        summary["probes"][probe.lower()] = {"per_prefix": per_prefix_list, "avg_flip_naive": avg}

    for d in (hp_dir, me_hp):
        json.dump(summary, open(d / "hallucination_summary.json", "w"), indent=2)

    print(f"[{model}/{dataset}] n={len(by_sample)}  records={len(recs)}")
    for probe in ("p1","p3","p4","p5","p6","p7"):
        if probe in summary["probes"]:
            v = summary["probes"][probe]
            if probe == "p1":
                avg = sum(d["flip_naive"] for d in v.values()) / len(v)
                print(f"   {probe}: avg flip {avg:.1f}%")
            else:
                print(f"   {probe}: avg {v['avg_flip_naive']:.1f}%")


def main():
    lookup = build_mumc_lookup()
    print(f"MUMC lookup: {len(lookup)} entries")
    for model in ("biomed_clip", "llava_med"):
        for dataset in ("vqa_rad", "vqa_med_2019", "vqa_med_2021"):
            process(model, dataset, lookup)


if __name__ == "__main__":
    main()
