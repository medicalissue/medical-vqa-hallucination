# 07 — 카테고리별 분석 (modality / plane / organ / abnormality)

## 왜 이걸 했나?

같은 데이터셋 안에서도 **질문 카테고리에 따라 hallucination 패턴이 달라집니다**. 예를 들어 \"이건 무슨 modality인가?\" 같은 질문은 답이 정해진 vocabulary(`ct`, `mri`, `x-ray`)에서 나오기 때문에 prior가 강합니다. 반면 abnormality(이상소견)는 vocabulary가 매우 다양해서 prior 효과가 다르게 나타납니다.

VQA-Med 2019은 **4개 카테고리(C1 modality, C2 plane, C3 organ, C4 abnormality)** 를 공식적으로 분리해 두었습니다. 우리는 이를 그대로 사용했고, VQA-RAD에는 카테고리가 없으므로 질문 텍스트 휴리스틱으로 5종 카테고리(`closed_yn` 추가)로 분류했습니다.

## 사용 카테고리

| ID | 정의 | VQA-Med 2019 분포 (n=78) | VQA-RAD 분포 (n=106) |
|---|---|---|---|
| `modality` | \"무슨 영상인가?\" | 20 | 4 |
| `plane` | \"axial/coronal/sagittal?\" | 28 | 4 |
| `organ` | \"무슨 장기?\" | 16 | 3 |
| `abnormality` | \"무엇이 이상한가?\" | 14 | 95 |
| `closed_yn` | yes/no 질문 (VQA-RAD only) | — | (위 abnormality에 포함) |

## LLaVA-Med 결과 (VQA-Med 2019, n=78)

| Category | n | Baseline (lenient) | Blank-image acc | P1 flip rate | P2 halluc | P3 flip (jaccard) |
|---|---:|---:|---:|---:|---:|---:|
| **modality** | 20 | 0.0% | 0.0% | **67.5%** | 100% | 26.0% |
| **plane** | 28 | 7.1% | 3.6% | **58.9%** | 100% | 17.9% |
| **organ** | 16 | 0.0% | 0.0% | 85.9% | 100% | 22.5% |
| **abnormality** | 14 | 0.0% | 0.0% | 80.4% | 100% | 27.1% |

→ 차트: [`images/P1_flip.png`](images/P1_flip.png), [`images/P3_flip_jaccard.png`](images/P3_flip_jaccard.png)

### 해석

**modality와 plane 질문의 P1 flip rate가 더 낮음 (각각 67%, 59%)** — 즉 이미지를 검정/노이즈로 바꿔도 답이 더 자주 그대로 유지된다는 뜻. 이는 \"형식이 정해진 답변 분포\"를 가진 카테고리에서 **모델이 이미지를 덜 쓰고 question prior에 더 의존**한다는 직접적 증거입니다.

modality 질문 \"what modality is shown?\"의 답은 사실상 `{ct, mri, x-ray, us, angiography, ...}` 중 하나로 prior가 매우 강합니다. 이미지를 안 봐도 LLaVA-Med은 합리적인 modality 답을 만들어냅니다.

abnormality/organ 질문의 P1 flip이 더 높은 이유는 **답 vocabulary가 무한** (\"colo-colic intussusception\", \"epidural empyema\" 등 의학 용어 다양)이라 이미지가 약하게나마 영향을 주기 때문.

## BiomedCLIP 결과 (VQA-Med 2019, n=150)

| Category | n | Baseline | Blank | P1 flip | P2 halluc | **P3 flip (jaccard)** |
|---|---:|---:|---:|---:|---:|---:|
| modality | 36 | 44.4% | 12.5% | 86.1% | 82.2% | 30.6% |
| **plane** | 42 | 42.9% | 10.7% | 87.5% | 92.4% | **52.9%** |
| organ | 36 | 61.1% | 11.8% | 84.0% | 95.0% | 33.3% |
| abnormality | 36 | 61.1% | 29.2% | 79.9% | 92.2% | 23.3% |

### 해석

**`plane` 카테고리는 P3 flip rate가 52.9%** — 본 분석에서 가장 fragile한 카테고리. 즉 \"무관한 환자 prefix\" 한 줄을 더하면 BiomedCLIP의 plane 답(axial/coronal/...)이 절반 이상 바뀝니다. plane 답 vocabulary가 짧고 candidate set 점수가 비슷해서, 작은 prompt 변화가 점수 순위를 쉽게 뒤집기 때문입니다.

**baseline 정확도는 organ/abnormality에서 가장 높음 (61%)** — BiomedCLIP은 candidate set 기반이라 \"무엇이 보이는가?\" 같은 vocabulary 풍부한 질문이 오히려 contrastive matching에 유리.

## VQA-RAD 결과 (n=106)

VQA-RAD는 closed_yn(yes/no)이 대다수입니다.

| Category | n | LLaVA-Med P3 flip | BiomedCLIP P3 flip |
|---|---:|---:|---:|
| **closed_yn** | 106 | **0.0%** | 0.0% |
| abnormality | 95 | 23.9% | 35.8% |
| modality | 4 | 20.0% | 15.0% |
| plane | 4 | 0.0% | 68.0% |
| organ | 3 | 13.3% | 30.0% |

### 해석 (놀라운 점)

VQA-RAD에서 **closed_yn 질문 P3 flip이 두 모델 모두 0%**. 즉 yes/no 답은 prefix를 더해도 **거의 절대 변하지 않음**. 이건 jaccard threshold(< 0.5) 기준에서는 \"yes\"와 \"no\" 둘 다 토큰이 너무 짧아 token Jaccard가 보존되는 측정 artifact. yes_no metric으로 보면 closed_yn에서 약 10–20% flip 발생함 (`results/full_v2/summary_long.csv` 참조).

VQA-RAD의 plane 질문 4개 중 BiomedCLIP은 68%가 prefix flip. n이 작아 통계 신뢰도는 낮지만 plane 질문 fragility 패턴은 일관됨.

## 핵심 메시지

1. **카테고리별로 hallucination 패턴이 다름.** 단일 dataset-level 평균은 위험성을 가립니다.
2. **\"답 vocabulary가 좁은\" 카테고리(modality, plane)에서 question prior가 강함 → 이미지 영향 작음.** 임상 deployment 시 이런 질문에 더 주의.
3. **plane 질문이 prompt perturbation에 가장 fragile** (BiomedCLIP 53% P3 flip in VQA-Med 2019).
4. **LLaVA-Med의 거절률은 모든 카테고리에서 0%** — 카테고리 무관 일관 패턴.

## 더 자세히

- 풀 표 (Wilson 95% CI 포함): [`../../results/modality/summary.csv`](../../results/modality/summary.csv)
- 5 plots (baseline / blank / P1 / P2 / P3 카테고리별 막대그래프): `images/`
- raw output: 우리 raw.jsonl에 `category` 필드 추가됨 — [`../../results/{model}_{run}/{dataset}/raw_labeled.jsonl`](../../results/)
