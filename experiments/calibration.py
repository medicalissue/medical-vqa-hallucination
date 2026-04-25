"""P6: Confidence calibration on closed-form (yes/no) VQA-RAD.

For BiomedCLIP we have explicit confidence (top_prob from softmax over
candidates). For LLaVA-Med we approximate confidence as 1.0 (always confident,
since the only signal we have is the binary correctness of the generated
text).

Outputs ECE (10-bin) and Brier on the closed-form orig variant of P1_blank.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metrics import expected_calibration_error, brier_score, _contains_answer


def calibration_from_jsonl(jsonl_path: Path, model_name: str):
    recs = [json.loads(l) for l in open(jsonl_path)]
    closed = [r for r in recs
              if r["probe"] == "P1_blank" and r["variant"] == "orig"
              and r["type"] == "closed"]
    if not closed: return {"ok": False, "reason": "no closed orig records"}
    correct = np.array([_contains_answer(r["pred"], r["gt"]) for r in closed], dtype=float)
    confs = np.array([r.get("confidence") if r.get("confidence") is not None else 0.5
                      for r in closed], dtype=float)
    return {
        "ok": True,
        "model": model_name,
        "n": int(len(closed)),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confs.mean()),
        "ECE": expected_calibration_error(confs, correct),
        "Brier": brier_score(confs, correct),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = calibration_from_jsonl(Path(args.jsonl), args.model_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
