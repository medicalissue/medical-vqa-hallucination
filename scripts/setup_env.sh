#!/usr/bin/env bash
# Run on a fresh AWS Deep Learning OSS PyTorch 2.7 AMI (Ubuntu 22.04, g5.xlarge)
set -euo pipefail

source /opt/pytorch/bin/activate

pip install --quiet \
    "transformers>=4.38,<4.50" \
    datasets huggingface_hub bitsandbytes accelerate \
    pillow opencv-python-headless matplotlib seaborn pandas tqdm scikit-learn \
    sentencepiece protobuf einops open_clip_torch ftfy

mkdir -p ~/mmbert_work ~/hf_cache
export HF_HOME=/home/ubuntu/hf_cache
python - <<'PY'
import torch, transformers, datasets
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
print(f"transformers={transformers.__version__} datasets={datasets.__version__}")
PY
