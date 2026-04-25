# 06 — 종합 해석 / 한계 / 시사점 (PPT 2–3 슬라이드)

## 슬라이드 1: 종합 메시지 — 5가지 발견

1. **\"보지만 듣지는 않는다\"** — Generative 모델 LLaVA-Med은 검정/흰색/노이즈 이미지에서도 baseline과 비슷한 정확도. 픽셀이 답에 거의 영향 안 줌.

2. **거절(refusal) 행동의 부재 — 가장 위험한 패턴** — LLaVA-Med은 명백한 image-text mismatch에 100% 답함. \"잘 모르겠습니다\" 류 출력 0%.

3. **텍스트 prefix에 fragility** — 무관한 환자 narrative 한 문장으로 약 18-45% 답이 의미상 변함.

4. **Demographic bias** — 의학적으로 무관한 성별·연령·인종·종교 prefix만으로 답이 흔들림. VQA-Med 2021에서 LLaVA-Med 35%.

5. **Metric 선택의 중요성** — naive bit-exact는 generative 모델을 unfairly 손해 보게 함. 4가지 metric 동시 보고 필수.

## 슬라이드 2: 임상 deployment에서 무엇을 의미하는가?

### 안전성 위험

- **잘못된 질문에 \"잘 모르겠다\"고 답하지 않는다** → 의료 환경에서 가장 위험. 의료진/환자가 잘못된 질문을 할 때 *그럴듯한 거짓*을 받음.
- **chart note 자동 결합 시 답이 바뀐다** → 환자 history를 prompt로 결합하는 임상 시스템에 spurious correlation 위험.
- **demographic 정보가 진단에 영향** → fairness audit 필수 영역.

### 권장사항

| 권장사항 | 누구를 위해 |
|---|---|
| 거절 행동 fine-tuning (RLHF \"don't know\" reward) | 모델 개발자 |
| Visual grounding consistency loss (blank image penalty) | 모델 개발자 |
| Prompt-level demographic bias audit | 의료 AI 도입 측 |
| Image-text consistency check (out-of-scope detection) | 의료 AI 도입 측 |
| 다중 metric 평가 의무화 | 모델 평가 측 |

## 슬라이드 3: 한계 + 향후 작업

### 한계

| 한계 | 영향 |
|---|---|
| 샘플 크기 (LLaVA n=20-106) | 절대 수치는 ±5%p 변동 가능. 트렌드는 stable. |
| `lenient` accuracy false positive | 통계적으로는 작음. yes_no metric으로 보완. |
| BiomedCLIP candidate set에 GT 포함 (VQA-Med 2021) | baseline 82.7%은 artifact. perturbation trend는 robust. |
| Refusal 키워드 휴리스틱 | LLaVA-Med은 어떤 거절 키워드도 출력 안 해 의미 없음. |
| 일반 도메인 sentence embedder | 의료 동의어 처리는 PubMedBERT 기반 embedder가 더 정확. |
| MedVInT은 구현 복잡성으로 제외 | 차후 비교군으로 추가 예정. |
| Probe set은 illustrative | 임상의 큐레이션 set으로 갱신 필요. |

### 향후 작업

- MedVInT-TE 통합 (PMC-VQA 저자 코드 기반)
- LLaVA-Med token-level log-prob → ECE/Brier 측정
- 의료 도메인 sentence embedder로 P3/P4 재계산
- Chain-of-thought \"don't know\" prompt ablation
- 임상의 큐레이션 mismatch 질문 set
- SLAKE, PathVQA 등 다른 데이터셋 generalization

## 풀 리포트와 참고

- **한국어 풀 리포트**: [`../results/REPORT_KO_v3.md`](../../results/REPORT_KO_v3.md) (337 lines)
- **GitHub repo**: <https://github.com/medicalissue/medical-vqa-hallucination>
- **모든 raw output**: `../../results/{biomed_clip,llava_med}_{big,combined}/{dataset}/raw.jsonl`
- **Wilson CI 포함 long-form CSV**: `../../results/full_v2/summary_long.csv` (306 rows)
