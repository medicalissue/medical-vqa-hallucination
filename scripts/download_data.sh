#!/usr/bin/env bash
# Download the three datasets into ~/mmbert_work/data
set -euo pipefail

source /opt/pytorch/bin/activate
export HF_HOME=${HF_HOME:-/home/ubuntu/hf_cache}
DATA=/home/ubuntu/mmbert_work/data
mkdir -p "$DATA"/{vqa_rad,vqa_med_2019,vqa_med_2021}

cd "$DATA/vqa_rad"
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("flaviagiammarino/vqa-rad")
ds.save_to_disk("/home/ubuntu/mmbert_work/data/vqa_rad/hf")
print({k: len(v) for k, v in ds.items()})
PY

cd "$DATA/vqa_med_2019"
# Zenodo record 10499039 — VQA-Med 2019 (Ben Abacha et al.)
wget -q --show-progress 'https://zenodo.org/records/10499039/files/VQAMed2019.zip?download=1' -O vqamed2019.zip
unzip -q -o vqamed2019.zip
rm -f vqamed2019.zip
ls -la

cd "$DATA/vqa_med_2021"
# Validation + test-with-GT from abachaa/VQA-Med-2021
wget -q --show-progress 'https://raw.githubusercontent.com/abachaa/VQA-Med-2021/main/Task1-VQA-2021-TestSet-w-GroundTruth.zip' -O test_gt.zip
wget -q --show-progress 'https://raw.githubusercontent.com/abachaa/VQA-Med-2021/main/VQA-Med-2021-Tasks-1-2-NewValidationSets.zip' -O val.zip
unzip -q -o test_gt.zip && unzip -q -o val.zip
rm -f test_gt.zip val.zip
ls -la

echo "=== DONE ==="
du -sh /home/ubuntu/mmbert_work/data/*
