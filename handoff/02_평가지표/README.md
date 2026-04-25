# 02 — 평가지표: \"어떻게 측정했는가?\"

## 왜 단일 metric으로 충분치 않은가?

LLaVA-Med 같은 generative 모델은 같은 sample에 대해 **표현이 매번 조금씩 다른 문장**을 출력합니다:

```
원본 이미지 →     \"Yes, the lesion appears to be wedge-shaped.\"
검정 이미지 → \"Yes, this lesion is wedge-shaped.\"
```

의미는 동일하지만, 단순 문자열 비교(`bit-exact`)로는 \"답이 바뀜\"으로 카운트됩니다.

→ **단일 metric만 쓰면 generative 모델의 진짜 행동이 가려집니다.** 본 분석은 4가지 metric을 모두 보고합니다.

→ 차트: [`images/P3_metrics_compare.png`](images/P3_metrics_compare.png) — 동일 P3 flip rate가 metric별로 30% → 18%로 변하는 것을 한 화면에서 확인

## 4가지 비교 metric 정의

### Flip rate (\"답이 바뀌었는가?\") 측정

| 이름 | 정의 | 강점 | 약점 |
|---|---|---|---|
| `naive` | 정규화 후 문자열 불일치 | 명확 | generative 표현 차이에 false flip |
| `yes_no` | 첫 yes/no 토큰 변화 (closed sample만) | 가장 fair | open question 분석 불가 |
| `jaccard` | 토큰 set Jaccard < 0.5 | dependency-free, 의미 overlap 반영 | threshold 임의성 |
| `embedding` | sentence-BERT cosine < 0.85 | semantic equivalence를 가장 잘 포착 | model 의존성 |

사용 모델: `sentence-transformers/all-MiniLM-L6-v2` (384차원)

### Accuracy (\"GT와 일치하는가?\") 측정

| 이름 | 정의 | 사용 시점 |
|---|---|---|
| `strict` | 정규화 후 정확 일치 | yes/no 같은 짧은 답에 사용 |
| `lenient` | GT 단어가 pred에 substring으로 포함 | 풍부한 generative 답에 GT가 들어 있으면 OK |
| `yes_no` | 첫 yes/no 토큰 비교 (closed only) | 가장 fair한 정확도 |
| `jaccard` | 토큰 Jaccard ≥ 0.3 | semantic match |

### Refusal rate

키워드 기반: \"cannot\", \"unable\", \"unclear\", \"insufficient\", \"need more\", \"i don\", \"not sure\" 등 12개 키워드.

## 신뢰구간

비율 추정에 **Wilson 95% interval**을 사용. n이 작거나 p가 0/1에 가까운 경우 정규근사가 무효해서.

표에 모두 [low, high] 형태로 같이 표시됩니다.

## 핵심 메시지

> **\"naive metric으로만 비교하면 LLaVA-Med을 unfairly 손해 보게 됩니다.\"**  
> P3 flip rate (irrelevant prefix) 예시:
> - naive: LLaVA 45% vs BiomedCLIP 45% → 비슷해 보임
> - **embedding: LLaVA 25% vs BiomedCLIP 28% → 비슷**
> - **yes_no: LLaVA 10% vs BiomedCLIP 20% → LLaVA가 더 안정적**
>
> 한 metric만 보면 잘못된 결론에 도달할 수 있습니다.
