# Medical VQA Hallucination Analysis

Reproducible experiments probing hallucination behaviors in medical Visual Question Answering (VQA) models.

## Motivation

We originally set out to run [MMBERT (Khare et al., 2021)](https://arxiv.org/abs/2104.01394) from the [official repo](https://github.com/virajbagal/mmbert), but no pretrained weights are publicly released. We therefore pivot to two **publicly-weighted** medical VQA models and run the same hallucination probes on both:

- **LLaVA-Med v1.5 (7B)** — `microsoft/llava-med-v1.5-mistral-7b` — generative
- **MedVInT-TE** — `xmcmic/MedVInT-TE` — classification/generation hybrid trained on PMC-VQA

## Datasets

| Dataset | Source | Size | License |
|---|---|---|---|
| VQA-RAD | HF `flaviagiammarino/vqa-rad` | 314 imgs, 2244 QA | CC0 |
| VQA-Med 2019 | [Zenodo 10499039](https://zenodo.org/records/10499039) | 3200+4500 imgs | research |
| VQA-Med 2021 (test) | [abachaa/VQA-Med-2021](https://github.com/abachaa/VQA-Med-2021) | 500 imgs w/ GT | research |

## Hallucination probes

1. **Blank image** — black / white / gaussian noise; keep original question
2. **Image-text mismatch** — wrong-organ questions against a fixed image
3. **Irrelevant patient prefix** — inject unrelated patient narrative
4. **Demographic perturbation** — vary gender/age/race prefix for same image+question
5. **Attention maps** — Grad-CAM (vision) + cross-attention visualization
6. **Confidence calibration** — ECE, Brier, entropy on perturbed inputs

## Hardware

AWS EC2 `g5.xlarge` (NVIDIA A10G 24GB) running Deep Learning OSS PyTorch 2.7 AMI (Ubuntu 22.04).

## Reproducing

See `scripts/setup_env.sh`, `scripts/download_data.sh`, and `experiments/run_all.sh`.
