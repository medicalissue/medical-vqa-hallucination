# 01 — 배경과 데이터셋

## 왜 MMBERT가 아니고 다른 모델로 했는가?

당초 계획은 **MMBERT (Khare et al., ISBI 2021)** 재현이었습니다. ROCO 의료 데이터로 pretraining한 multimodal BERT 방법.

문제: **공식 GitHub repo에 사전학습 가중치가 공개되어 있지 않습니다.**

- 모든 체크포인트 경로가 저자 로컬 경로 `/home/viraj.bagal/...`로 하드코딩
- HuggingFace · Google Drive · GitHub Releases 어디에도 가중치 없음
- Issue #2 (가중치 공개 문의)에 답변 없음
- `eval.py`도 \"inference-only\"가 아니라 fine-tuned classification head를 요구 — 가중치 없으면 추론 불가

→ **공개 가중치가 있는 두 모델로 전환:**

## 사용한 두 모델

의료 VQA의 두 주요 패러다임을 대표하는 모델을 각각 한 종씩 선택했습니다.

| 모델 | 패러다임 | 출력 형태 | 모델 ID | 가중치 크기 |
|---|---|---|---|---|
| **LLaVA-Med v1.5 (7B)** | Generative LLM | 자유 문장 (\"Yes, the lesion appears...\") | `chaoyinshe/llava-med-v1.5-mistral-7b-hf` | ~14 GB |
| **BiomedCLIP** | Contrastive (Vision-Language) | candidate set에서 최고 점수 답 | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | ~500 MB |

LLaVA-Med은 CLIP ViT-L/14 + Mistral 7B, instruction-tuned on PMC-15M.  
BiomedCLIP은 ViT-B/16 + PubMedBERT-256, contrastive on PMC-15M. zero-shot 사용.

## 사용한 세 데이터셋

| 데이터셋 | 출처 | 본 분석 사용 분량 | 라이선스 |
|---|---|---|---|
| **VQA-RAD** | HF `flaviagiammarino/vqa-rad` | BiomedCLIP n=150, LLaVA n=106 | CC0 |
| **VQA-Med 2019** | Zenodo 10499039 (ImageCLEF 2019 챌린지) | BiomedCLIP n=150, LLaVA n=54 | CC-BY-4.0 |
| **VQA-Med 2021** | abachaa/VQA-Med-2021 (ImageCLEF 2021) | BiomedCLIP n=150, LLaVA n=20 | research |

VQA-RAD = 의료진 작성 yes/no 위주 (closed). VQA-Med 2019/2021 = ImageCLEF 챌린지용, 의학 용어 명사구 답 (open).

## 인프라

- AWS EC2 **g5.xlarge** (NVIDIA A10G 24GB, us-west-2)
- Deep Learning OSS PyTorch 2.7 AMI (Ubuntu 22.04)
- 총 비용: 약 $5–8 (인스턴스 가동 5–8시간)
