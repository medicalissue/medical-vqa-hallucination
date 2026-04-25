# 의료 VQA 할루시네이션 분석 — 자료 폴더

## 이 폴더가 뭐야?

분석 결과를 **단계별로 따라갈 수 있게** 정리한 자료입니다. 각 폴더의 `README.md`만 읽어도 그 섹션은 끝.

## 폴더 구성

| # | 폴더 | 무엇을 다루나 |
|---|---|---|
| 0 | `00_요약/` | 한 페이지 결론. 가장 중요한 4개 차트만. |
| 1 | `01_배경과_데이터셋/` | \"왜 이걸 했는가\" + 사용한 모델/데이터셋 표 |
| 2 | `02_평가지표/` | 4가지 metric 정의. **\"naive bit-exact는 generative 모델에 unfair\"라는 메시지가 가장 중요**. |
| 3 | `03_데이터셋별_결과/` | 데이터셋마다 결과 표 + 차트 |
| 4 | `04_모델비교/` | LLaVA-Med vs BiomedCLIP 정면 비교 |
| 5 | `05_사례_분석/` | raw output 가장 인상적인 사례들 |
| 6 | `06_종합_해석/` | 한계, 시사점, 향후 작업 |

## 어디서부터 읽어야 할지 모르겠으면

1. `00_요약/README.md` — 5분 안에 전체 그림.
2. `04_모델비교/README.md` — \"LLaVA-Med이 더 안전한가, BiomedCLIP이 더 안전한가?\"의 답.
3. `05_사례_분석/README.md` — 가장 임팩트 큰 \"raw output\".

## 모든 plot은 어디?

- 각 폴더의 `images/` 안에 그 섹션과 관련된 plot이 모여 있습니다.
- 모든 원본 plot은 `../results/full_v2/plots/` 에 (handoff에는 가장 중요한 것만 골라 복사).

## 풀 리포트 (학술적 톤)는?

- 한국어 풀 리포트: `../results/REPORT_KO_v3.md` (337 lines, methodology · CI · 사례)
- 영문 README: `../README.md`
- raw output: `../results/{biomed_clip,llava_med}_{big,combined}/{dataset}/raw.jsonl`
- GitHub: <https://github.com/medicalissue/medical-vqa-hallucination>

## 데이터 마감 기준

- BiomedCLIP: 데이터셋당 n=150
- LLaVA-Med: vqa_rad n=106, vqa_med_2019 n=54, vqa_med_2021 n=20 (sweep 진행 중. 완료 시 자동 갱신)
- 작성일: 2026-04-25
