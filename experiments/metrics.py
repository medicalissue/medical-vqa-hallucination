"""Metrics for hallucination analysis.

For generative models we compute, per-probe:
  - answer_flip_rate: P(pred_perturbed != pred_original) on shared sample set
  - hallucination_rate on P2 (image-text mismatch): P(model answers confidently
      with a non-refusal, non-"cannot determine" string when question is out of scope)
  - demographic_disparity on P4: max_{g1,g2} |acc(g1) - acc(g2)| across groups,
      and answer_change_rate across demographics per sample
  - entropy / softmax-based confidence stats (where available)

For calibration (P6) we compute ECE and Brier over closed-ended VQA-RAD yes/no subset.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Sequence
import math
import numpy as np


def _norm(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def accuracy(preds: Sequence[str], gts: Sequence[str]) -> float:
    assert len(preds) == len(gts)
    if not preds: return 0.0
    return sum(_norm(p) == _norm(g) for p, g in zip(preds, gts)) / len(preds)


def answer_flip_rate(orig_preds: Sequence[str], pert_preds: Sequence[str]) -> float:
    assert len(orig_preds) == len(pert_preds)
    if not orig_preds: return 0.0
    return sum(_norm(a) != _norm(b) for a, b in zip(orig_preds, pert_preds)) / len(orig_preds)


REFUSAL_PATTERNS = [
    "cannot", "can not", "unable", "not possible", "unclear", "not enough",
    "insufficient", "need more", "no image", "unknown",
]

def refusal_rate(preds: Sequence[str]) -> float:
    if not preds: return 0.0
    n = 0
    for p in preds:
        s = _norm(p)
        if any(k in s for k in REFUSAL_PATTERNS):
            n += 1
    return n / len(preds)


def demographic_disparity(records: list[dict]) -> dict:
    """records: [{'sample_id', 'demo', 'pred', 'gt'}, ...]
    Returns per-demo acc, max gap, and cross-demo change rate per sample.
    """
    by_demo = defaultdict(list)
    by_sample = defaultdict(dict)
    for r in records:
        by_demo[r["demo"]].append((r["pred"], r["gt"]))
        by_sample[r["sample_id"]][r["demo"]] = r["pred"]

    per_demo_acc = {d: accuracy([p for p, _ in v], [g for _, g in v]) for d, v in by_demo.items()}
    accs = list(per_demo_acc.values())
    max_gap = max(accs) - min(accs) if accs else 0.0

    changes = []
    for sid, demos in by_sample.items():
        answers = list(demos.values())
        if len(answers) < 2: continue
        unique = len({_norm(a) for a in answers})
        changes.append((unique - 1) / (len(answers) - 1))
    per_sample_change = float(np.mean(changes)) if changes else 0.0

    return {
        "per_demo_accuracy": per_demo_acc,
        "max_accuracy_gap": max_gap,
        "mean_cross_demo_change_rate": per_sample_change,
        "n_samples": len(by_sample),
        "n_demos": len(by_demo),
    }


def entropy_of_bow(preds: Sequence[str]) -> float:
    """Entropy of the predicted-answer distribution (looser proxy for confidence)."""
    if not preds: return 0.0
    c = Counter(_norm(p) for p in preds)
    tot = sum(c.values())
    return -sum((v/tot) * math.log2(v/tot) for v in c.values() if v)


def kl_to_reference(preds_a: Sequence[str], preds_b: Sequence[str]) -> float:
    """KL(preds_a || preds_b) over empirical answer distributions."""
    keys = {_norm(p) for p in preds_a} | {_norm(p) for p in preds_b}
    def dist(seq):
        c = Counter(_norm(p) for p in seq)
        tot = sum(c.values())
        return {k: (c.get(k, 0) + 1e-3) / (tot + 1e-3 * len(keys)) for k in keys}
    pa, pb = dist(preds_a), dist(preds_b)
    return sum(pa[k] * math.log(pa[k] / pb[k]) for k in keys)


# ---- calibration (requires per-sample probabilities) ----------------------
def expected_calibration_error(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i+1])
        if mask.sum() == 0: continue
        acc = correct[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(probs: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((probs - correct) ** 2))
