"""Unified loader for VQA-RAD, VQA-Med 2019, VQA-Med 2021.

Each loader returns a list of dicts:
  {sample_id, image (PIL), question, answer, type, dataset}
where `type` is "closed" (yes/no answer) or "open".
"""
from __future__ import annotations
import random
from pathlib import Path
from PIL import Image

DATA = Path("/home/ubuntu/mmbert_work/data")


def _close_or_open(ans: str) -> str:
    return "closed" if str(ans).strip().lower() in ("yes", "no") else "open"


def load_vqa_rad(n: int, seed: int = 0):
    from datasets import load_from_disk
    ds = load_from_disk(str(DATA / "vqa_rad" / "hf"))["test"]
    idx = list(range(len(ds))); random.Random(seed).shuffle(idx)
    closed = [i for i in idx if str(ds[i]["answer"]).lower() in ("yes", "no")]
    opened = [i for i in idx if i not in closed]
    sel = []
    for i in closed[: n // 2]:
        sel.append({"sample_id": f"rad_{i}", "image": ds[i]["image"],
                    "question": ds[i]["question"], "answer": ds[i]["answer"],
                    "type": "closed", "dataset": "vqa_rad"})
    for i in opened[: n - len(sel)]:
        sel.append({"sample_id": f"rad_{i}", "image": ds[i]["image"],
                    "question": ds[i]["question"], "answer": ds[i]["answer"],
                    "type": "open", "dataset": "vqa_rad"})
    return sel


def load_vqa_med_2019(n: int, seed: int = 0):
    """Use the 2019 official test set (500 imgs, 500 QA, w/ reference answers)."""
    base = DATA / "vqa_med_2019" / "VQAMed2019Test"
    qa_file = base / "VQAMed2019_Test_Questions_w_Ref_Answers.txt"
    img_dir = base / "VQAMed2019_Test_Images"
    rows = []
    for line in qa_file.read_text().splitlines():
        if not line.strip(): continue
        parts = line.split("|")
        if len(parts) < 4: continue
        # format: synpic_id|category|question|answer
        sid, cat, q, a = parts[0], parts[1], parts[2], "|".join(parts[3:])
        img = img_dir / f"{sid}.jpg"
        if not img.exists(): continue
        rows.append({"sample_id": f"med19_{sid}_{cat}", "image_path": str(img),
                     "question": q, "answer": a, "type": _close_or_open(a),
                     "dataset": "vqa_med_2019", "category": cat})
    random.Random(seed).shuffle(rows)
    sel = []
    for r in rows[:n]:
        r["image"] = Image.open(r["image_path"]).convert("RGB")
        del r["image_path"]
        sel.append(r)
    return sel


def load_vqa_med_2021(n: int, seed: int = 0):
    base = DATA / "vqa_med_2021" / "Task1-VQA-2021-TestSet-w-GroundTruth"
    q_file = base / "Task1-VQA-2021-TestSet-Questions.txt"
    a_file = base / "Task1-VQA-2021-TestSet-ReferenceAnswers.txt"
    img_dir = base / "VQA-500-Images"
    qs = {}
    for line in q_file.read_text().splitlines():
        if "|" not in line: continue
        sid, q = line.split("|", 1)
        qs[sid.strip()] = q.strip()
    ans_map = {}
    for line in a_file.read_text().splitlines():
        if "|" not in line: continue
        parts = line.split("|")
        sid = parts[0].strip()
        # the file has multiple acceptable answers; take the first
        a = parts[1].strip() if len(parts) > 1 else ""
        ans_map[sid] = a
    rows = []
    for sid, q in qs.items():
        if sid not in ans_map: continue
        img = img_dir / f"{sid}.jpg"
        if not img.exists(): continue
        rows.append({"sample_id": f"med21_{sid}", "image_path": str(img),
                     "question": q, "answer": ans_map[sid],
                     "type": _close_or_open(ans_map[sid]),
                     "dataset": "vqa_med_2021"})
    random.Random(seed).shuffle(rows)
    sel = []
    for r in rows[:n]:
        r["image"] = Image.open(r["image_path"]).convert("RGB")
        del r["image_path"]
        sel.append(r)
    return sel


LOADERS = {
    "vqa_rad": load_vqa_rad,
    "vqa_med_2019": load_vqa_med_2019,
    "vqa_med_2021": load_vqa_med_2021,
}


def load(dataset: str, n: int, seed: int = 0):
    return LOADERS[dataset](n, seed)
