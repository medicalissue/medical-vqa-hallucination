# MUMC-format Export

`analysis_mumc/` 형식과 컬럼 구조를 그대로 따른 CSV/JSON 파일들. 두 모델(LLaVA-Med, BiomedCLIP) × 세 데이터셋(VQA-RAD, VQA-Med 2019, VQA-Med 2021).

## 구조

```
mumc_export/
├── llava_med/
│   ├── eval_finetuned/
│   │   ├── vqa_rad/results_vqa_rad.csv
│   │   ├── med2019_local/results_med2019_local.csv
│   │   └── vqa_med2021/results_vqa_med2021.csv
│   └── model_response/
│       ├── vqa_rad/   {p1,p3,p4,hallucination_summary}
│       ├── med2019_local/  {p1,p3,p4,p7,hallucination_summary}
│       └── vqa_med2021/    {p1,p3,p4,p7,hallucination_summary}
└── biomed_clip/
    └── (same)
```

## CSV 컬럼 (MUMC와 동일)

### `eval_finetuned/{dataset}/results_{dataset}.csv`
```
qid, question, answer, modality, pred_label, correct
```
- `modality`: 질문+GT에서 휴리스틱으로 추론 (X-ray/CT/MRI/Ultrasound/Angiography/Mammography/Nuclear/unknown)
- `correct`: lenient match (GT phrase가 pred에 substring으로 등장하면 1)

### `model_response/{dataset}/p1_image_grounding.csv`
```
qid, question, modality, gt, pred_original,
pred_blank, flip_blank, pred_white, flip_white,
pred_noise, flip_noise, pred_gray, flip_gray
```
- `flip_*` = 1 if `pred_*`가 `pred_original`과 정규화 후 다름

### `model_response/{dataset}/p3_irrelevant_text.csv`
```
qid, question, modality, gt, prefix_id, prefix_text, pred_original, pred_perturbed, flip
```
- prefix가 5종 (irrelevant patient narrative)이라 sample당 5 row

### `model_response/{dataset}/p4_demographic.csv`
같은 스키마. prefix가 11종 (성별·연령·인종·종교).

### `model_response/{dataset}/p7_modality_mismatch.csv`
같은 스키마. prefix가 false modality framing.
- VQA-RAD에는 P7 sweep 안 했으므로 파일 없음
- VQA-Med 2019/2021은 P7 결과 있음

## `hallucination_summary.json` 형식

```json
{
  "dataset": "vqa_med2021",
  "model": "llava_med",
  "n_samples": 78,
  "probes": {
    "p1": { "blank": {n_total, flip_naive, n_yes_no, flip_yes_no}, "white": ..., "noise": ..., "gray": ... },
    "p3": { "per_prefix": [...], "avg_flip_naive": ... },
    "p4": { "per_prefix": [...], "avg_flip_naive": ... },
    "p7": { "per_prefix": [...], "avg_flip_naive": ... }
  }
}
```

`flip_naive` = 정규화 문자열 비교 기준 % (MUMC와 동일)
`flip_yes_no` = closed yes/no sample만 본 % (MUMC와 동일)

## 차이점 / 주의점

| 항목 | MUMC | 우리 |
|---|---|---|
| n_samples | dataset당 500 | LLaVA 78–106, BiomedCLIP 150 |
| modality 라벨 | 일부 명시 (MRI/unknown), 일부 unknown | 휴리스틱 자동 추론 |
| P5 (medical history), P6 (socioeconomic) | 있음 | 없음 (probe 정의 안 함) |
| P7 prefix | \"This image was obtained using {mod}.\" 5종 | 우리 P7는 \"What abnormality is seen on this {mod} image?\" 등 3 question × 5 false_mod = 15 variants per sample |
| BiomedCLIP의 P7 | — | candidate scoring이라 prefix를 출력에 채택 못 함 (구조적). flip이 거의 0 |

## 모델별 baseline 정확도

`results_*.csv`에서 `correct` 컬럼 평균:

| Model | VQA-RAD | VQA-Med 2019 | VQA-Med 2021 |
|---|---|---|---|
| LLaVA-Med | 9.4% | 1.9% | 0.0% |
| BiomedCLIP | 33.3% | 52.0% | 82.7% |

> BiomedCLIP의 VQA-Med 2021 82.7%는 candidate set에 GT가 포함된 artifact임 (perturbation trend는 robust).

## 데이터 출처

원본 raw output: `../{model}_{combined|big}/{dataset}/raw.jsonl` 와 `../p7_{model}_big/raw.jsonl`. MUMC 형식으로 변환한 스크립트: `../../scripts/export_mumc_format.py`.
