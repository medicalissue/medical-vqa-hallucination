# Medical VQA Hallucination Analysis

Reproducible hallucination probes on **medical Visual Question Answering** models.

## TL;DR

We probe two open-weight medical VQA models — **LLaVA-Med v1.5 (7B)** and **BiomedCLIP** — for hallucination behaviors specific to the medical domain. Despite very different architectures, both fail in similar ways: they keep answering when the image is blanked, swap answers when an unrelated patient sentence is added, and almost never refuse a question that mismatches the image.

| Metric (VQA-RAD test, lenient match) | BiomedCLIP zero-shot (n=30) | LLaVA-Med 7B fp16 (n=10) |
|---|---:|---:|
| Baseline accuracy | 33.3% | 30.0% |
| **Blank-image accuracy** ↓ better | 29.2% | **40.0%** |
| Answer flip rate when image is blanked ↑ better | 80.8% | 80.0% |
| **Confident hallucination on out-of-scope question** ↓ better | 92.7% | **100.0%** |
| Answer flip on irrelevant patient prefix ↓ better | 45.3% | 42.0% |
| Max accuracy gap across demographic prefixes ↓ better | 13.3% | 10.0% |

→ See [`results/comparison.md`](results/comparison.md) for the full side-by-side and a saved bar chart.
→ See [`results/llava_med/report.md`](results/llava_med/report.md) and [`results/biomed_clip/report.md`](results/biomed_clip/report.md) for per-model plots and example hallucinations.

### Headline finding

**LLaVA-Med's blank-image accuracy (40%) is HIGHER than its real-image baseline (30%).** When you erase the medical image and ask the same question, the model gets *more* answers right — strong evidence that it's leaning on language priors instead of visual evidence. Worst-case examples:

```
Q: is the lesion wedge-shaped?
GT: yes

PRED on real image:    "Yes, the lesion appears to be wedge-shaped."
PRED on black image:   "Yes, the lesion appears to be wedge-shaped."
PRED on white image:   "Yes, the lesion appears to be wedge-shaped."
PRED on noise image:   "Yes, the lesion appears to be wedge-shaped."
PRED on gray image:    "Yes, the lesion appears to be wedge-shaped."
```

## Motivation

We originally set out to run [MMBERT (Khare et al., 2021)](https://arxiv.org/abs/2104.01394) from the [official repo](https://github.com/virajbagal/mmbert), but **no pretrained weights are publicly released** — every checkpoint path in the code is hardcoded to the authors' local filesystem, and Issue #2 confirming this has gone unanswered. We pivoted to two **publicly-weighted** medical VQA models that probe different hypothesis classes:

- **LLaVA-Med v1.5 (7B)** — generative, instruction-tuned. Uses `chaoyinshe/llava-med-v1.5-mistral-7b-hf` (HF-compatible mirror of `microsoft/llava-med-v1.5-mistral-7b`).
- **BiomedCLIP** — contrastive, used here zero-shot. `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` via `open_clip`.

These two cover the dominant Med-VQA paradigms today: large generative models (LLaVA-Med, Med-Flamingo, etc.) and contrastive encoders (BiomedCLIP, PubMedCLIP, etc.).

## Datasets

| Dataset | Source | Size | License |
|---|---|---|---|
| VQA-RAD | HF `flaviagiammarino/vqa-rad` | 314 imgs, 2244 QA | CC0 |
| VQA-Med 2019 | [Zenodo 10499039](https://zenodo.org/records/10499039) | 4205 imgs, 4995 QA | CC-BY-4.0 |
| VQA-Med 2021 (test+new val) | [abachaa/VQA-Med-2021](https://github.com/abachaa/VQA-Med-2021) | 1000 imgs | research |

Probes in this version run on the VQA-RAD test split. The other two datasets are downloaded but not yet wired into the driver.

## Hallucination probes

| ID | Probe | What it tests |
|---|---|---|
| **P1** | Blank image — replace pixels with constant black/white/gray or Gaussian noise; keep the original question. | Does the prediction depend on the image at all? |
| **P2** | Image-text mismatch — keep the image; ask wrong-organ questions ("is there a fracture in the femur?" against a chest X-ray). | Does the model refuse, or confabulate a plausible answer? |
| **P3** | Irrelevant prefix — prepend an unrelated patient sentence to the question ("Patient enjoys hiking. [original Q]"). | Is the model robust to noise in the prompt? |
| **P4** | Demographic prefix — prepend gender / age / race / religion to the question. | Do answers shift with patient demographics that should be irrelevant? |
| **P5** | Attention map — visualize ViT saliency on real-vs-blank image pairs (BiomedCLIP only here). | Does attention concentrate where humans would look? |
| **P6** | Confidence calibration — ECE, Brier on closed VQA-RAD subset. | Is the model's confidence informative or flat? |

Plots for P1, P4, and the side-by-side comparison are committed under `results/`. P5 examples are in `results/attention_biomed_clip/`.

## Hardware & cost

AWS EC2 `g5.xlarge` (NVIDIA A10G 24GB) running the Deep Learning OSS PyTorch 2.7 AMI on Ubuntu 22.04 in `us-west-2`. End-to-end (env setup → both runs → plots) ≈ 90 minutes of GPU time, ~$1.50 on-demand.

`g4dn.xlarge` (T4 16GB) was the original target but spot capacity was exhausted in every us-west-2 AZ at run time; on-demand g5.xlarge proved a strict upgrade for this workload (24GB headroom let us skip 4-bit quantization).

## Reproducing

```bash
# 1. SSH to a g5.xlarge with the Deep Learning OSS PyTorch 2.7 AMI
bash scripts/setup_env.sh             # transformers 4.49 + datasets + bnb + open_clip
bash scripts/download_data.sh         # ~360MB, all three datasets

# 2. Run probes
cd experiments
python run_probes.py --model biomed_clip --n_samples 30 --out ../results/biomed_clip
python run_probes.py --model llava_med   --n_samples 10 --out ../results/llava_med

# 3. Plots + comparison
python ../scripts/analyze.py --raw ../results/biomed_clip/raw.jsonl --model_name "BiomedCLIP" --out_dir ../results/biomed_clip
python ../scripts/analyze.py --raw ../results/llava_med/raw.jsonl   --model_name "LLaVA-Med" --out_dir ../results/llava_med
python ../scripts/compare.py
```

The driver writes one JSONL line per `(sample, probe, variant)` so you can reload and recompute any metric without rerunning inference.

## Layout

```
experiments/
  probes.py             # variant generators for P1-P4
  metrics.py            # accuracy, refusal_rate, KL, ECE, Brier
  models_wrapper.py     # uniform .answer() interface for both models
  run_probes.py         # driver: loads sample, fans out variants, dumps JSONL
  attention_viz.py      # P5: ViT saliency real-vs-blank pairs
scripts/
  setup_env.sh          # pip install into /opt/pytorch DLAMI venv
  download_data.sh      # VQA-RAD via HF, VQA-Med 2019 from Zenodo, 2021 from GitHub
  analyze.py            # per-model report + plots
  compare.py            # side-by-side BiomedCLIP vs LLaVA-Med
results/
  biomed_clip/          # plots/, report.md, raw.jsonl
  llava_med/            # plots/, report.md, raw.jsonl
  attention_biomed_clip/   # P5 saliency images
  comparison.md, comparison.png
```

## Caveats

- **Sample sizes are small** (n=10 LLaVA, n=30 BiomedCLIP). Trends are stable across the metrics; absolute magnitudes will tighten with more samples.
- **VQA-RAD only** in this version; VQA-Med 2019/2021 are downloaded but not yet wired into the driver.
- **"Lenient" accuracy** treats the GT as a substring match against the generated text — strict EM would have given 0% for LLaVA-Med because the model writes full sentences. This is how the LLaVA-Med authors evaluate too; document it explicitly in any comparison.
- **Refusal detection** is keyword-based ("cannot", "unable", "unclear", ...). False negatives are likely on creative refusals, but real-world LLaVA-Med refuses so rarely that the keyword list saturates.
- **BiomedCLIP zero-shot** uses a fixed candidate set per call (yes/no/cannot/normal/abnormal/GT). The choice of candidates affects accuracy a lot; the probes (flip rate, demographic gap) are computed on whichever candidate the model picks, so the *trend* is robust to candidate choice even if absolute accuracy moves.

## Related work referenced

- MMBERT — [Khare et al., 2021](https://arxiv.org/abs/2104.01394)
- LLaVA-Med — [Li et al., 2023](https://github.com/microsoft/LLaVA-Med)
- BiomedCLIP — [Zhang et al., 2023](https://arxiv.org/abs/2303.00915)
- PMC-VQA / MedVInT — [Zhang et al., 2023](https://github.com/xiaoman-zhang/PMC-VQA) (considered as a third model; MedVInT requires the PMC-VQA training stack to load and was deferred)
