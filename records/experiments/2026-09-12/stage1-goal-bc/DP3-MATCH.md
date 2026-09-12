# MLP BC — Codex DP3와 같은 3시범 최소 비교

2026-09-13 00:05–00:08. 작은 자료에서의 **실행 재현 비교**이며 미지 물체 일반화 실험이 아니다. PPO는 보류. 기존 BC v1·v2 결과와
체크포인트는 보존했다. 결과를 보고 설정을 바꾸지 않았다.

## 같은 것 (Codex `stage1-dp3-v1`에서 그대로)

- 시범 3개(고정 선택 `selection.json`): AlphabetSoup `409a5d4a` 1024 tick · Peaches `9aa66578` 1047 · Raisins `0ef25c7b` 1018. 초기 상태·모델 동일.
- 관측: Codex `Archive.observation()`의 점군 640×3 + 상태 125, 과거 2프레임(DP3 `sequence_indices` padding). 학습 자료는 Codex가 저장한
  `run/features/*.npz`(읽기 전용). 정규화는 학습 자료만으로 limits([-1,1]) 계산.
- 행동: 27차원 최종 actuator 명령, ctrlrange clip만(보상·접촉 제어 추가 없음).
- 학습: seed 42, batch 64, AdamW lr 1e-4, wd 1e-6, warmup 500 + cosine, grad clip 1, 6,000 갱신, episode 균등 샘플링.
- 평가: 마지막 체크포인트, 3조건 × seed 1000–1002, 원본 tick 길이 예산, 품질 위반 때만 조기 종료, 종료 시 `Outcome`(contact_acquisition_guard_v2) 판정.
- 코드: Codex 고정 사본의 `stage1_dp3`·`icra`(제 사본과 7개 파일 동일)를 import, 실행기 `scripts/stage1_bc/dp3_match.py`(lecun 고정 사본 `code-v6`).

## 다른 것 (구조상)

MLP 3×512(ELU) — diffusion U-Net 대신; tick마다 행동 1개 — DP3는 16개 묶음 중 8개 실행; EMA 없이 마지막 갱신.

## 결과

재생 대조군(학습 전 관문): 3/3 통과 — 매 tick 관측이 DP3 저장본과 **비트 동일**, 상태·신호 오차 0, 기록 명령의 phase-free 판정 3/3 성공.

학습 6,000 갱신, 최종 손실 9.8e-5(29 s, GPU 2), 체크포인트 sha256 `682733ce700eb755…`.

| 조건 | MLP BC 완주 | 최초 이탈 (|q − teacher| > 0.05 rad 팔 / 0.10 rad 손가락) | 종료 | DP3 (Codex) |
| --- | ---: | --- | --- | ---: |
| AlphabetSoup | 0/3 | tick 8 (팔 0.051) | tick 78 digit_force | 2/3 |
| Peaches | 0/3 | tick 47 (팔 0.051) | tick 106 table_collision | 0/3 |
| Raisins | 0/3 | tick 18 (팔 0.053, 손가락 0.043) | tick 210 table_collision, clip 94 tick | 3/3 |
| 합계 | **0/9** | | | **5/9** |

- MLP는 결정적이라 seed 3개의 결과가 같다(평가 예산은 DP3와 같게 9회).
- 세 조건 모두 접촉 획득 전(접근 단계)에 실패했다. 75시범 BC(v1·v2)에서 본 폐루프 이탈과 같은 양상이다.
- 이 차이가 모델 종류(MLP/diffusion)·행동 묶음 실행·EMA 중 무엇에서 오는지는 이 비교로 분리되지 않는다.

결과 파일: `lecun-results/runs/mlp-dp3match-v1/{config,replay,training-result,evaluation}.json` (SHA256SUMS). lecun 원본 폴더는 쓰기 금지.
