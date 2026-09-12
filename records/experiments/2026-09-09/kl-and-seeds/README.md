<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

**범위 정정:** 이 계열의 주 실험은 고정 후보의 privileged 팔 접근이다. 다양한 물체의 유형/후보 선정·시범 적응 RL이 실패했다는 결론은 아니다.

# 규칙 고정 후 본 실험: 학습 seed 3개, 선택 절차가 반환한 정책 비교

검토 의견의 보완 세 가지를 먼저 반영한 뒤 예비 실행 1회와 본 실험 3회를 돌렸다.
후보·장면·보상·게이트·충돌 제한은 고정, β=0, BC 초기화, warm-up 40 update.

## 0. 보완한 것

| 문제 | 조치 |
| --- | --- |
| KL이 `mean(old_logp − new_logp)`라 부호가 있는 표본 추정값 | 정책이 Gaussian이므로 **해석적 KL**로 교체. 상태별로 7개 행동 차원을 합산한 뒤 상태 평균. 합성 사례(평균 1σ 차이)에서 0.5를 돌려주는 것을 테스트로 고정 |
| 취소한 갱신도 `actor_updates`에 포함 | **시도·승인·취소**를 분리 집계. 체크포인트 번호와 진행 표는 승인된 갱신 기준 |
| 감시자가 실패해도 학습이 계속될 수 있음 | 후보 실패 시 **STOP을 먼저** 쓰고 BC 재평가는 그다음. 평가 예외도 중지 사유로 기록. 학습 생존은 **PID**로 확인(`trainer.pid`) |
| 캐시 키에 장면·manifest·환경 코드가 없음 | 본 실험은 **새 캐시**(`eval-cache-v2`) 사용 |

테스트 336 passed(감시 4개, 갱신 3개 추가).

## 1. 예비 실행 — KL 한계를 측정으로 정했다

별도 학습 seed 9, warm-up 40 + 갱신 시도 10회, 시작값 `--kl-limit 0.01`.

**시도 10회, 승인 0회, 취소 10회.** 즉 0.01은 정책을 전혀 바꾸지 못하는 설정이다.
실측 KL은 0.0108–0.0383(중앙값 0.0263). 이 분포에서 한계를 **0.03**으로 정했다
(예비 갱신의 6/10 승인). 본 실험에서는 바꾸지 않았다. 예비 seed 9는 본 실험
seed 0·1·2와 분리했다.

## 2. 본 실험 3회 — 학습과 중지

| seed | 스텝 | 시도/승인/취소 | 갱신 후 KL(최소/중앙/최대) | 중지 |
| ---: | ---: | --- | --- | --- |
| 0 | 74,752 | 33 / 31 / 2 | 0.0147 / 0.0168 / 0.0217 | 72,704 체크포인트 평균행동 실패(침투) → **감시자가 자동 중지** |
| 1 | 64,512 | 23 / 20 / 3 | 0.0152 / 0.0235 / 0.0274 | 53,248 실패(물체 밀림) → 자동 중지 |
| 2 | 102,400 | 60 / 43 / 17 | 0.0165 / 0.0226 / 0.0376 | 실패 없음, 예산 소진으로 정상 종료 |

세 번 모두 실패 시 BC 기준을 다시 확인했고 매번 "still completes"였다(환경 문제 아님).
**감시–중지 연결은 의도대로 작동한다**: 밤 실험에서는 첫 실패 후 수만 스텝을 더
돌았지만, 이번에는 seed 0이 실패 확인 후 2,048스텝, seed 1이 11,264스텝 만에 멈췄다.

취소율은 6% / 13% / 28%다. **"갱신이 거의 전부 취소돼 정책이 바뀌지 않은" 상황은
아니다**(예비 실행의 0.01은 그런 상황이었고, 그래서 그 값을 쓰지 않았다).

## 3. 선택 절차가 반환한 정책

개발 seed 100–119, std 0.05. 원본 BC를 후보에 포함하고, 평균 행동 완주 실패는 제외.

| seed | 후보별 개발 완주 | 반환 | 개선 |
| ---: | --- | --- | --- |
| 0 | BC 13, 40,960 13, 51,200 14, **61,440 20**, 72,704·73,728 평균행동 실패 | `step_00061440` | 예 |
| 1 | BC 13, 40,960 13, 53,248·64,512 평균행동 실패 | **BC** | 아니오 |
| 2 | BC 13, 40,960 13, 56,320 5, 57,344 5, 71,680 13, 83,968 13, **99,328 17** | `step_00099328` | 예 |

seed 2는 중간에 5/20까지 떨어졌다가 회복해 마지막 체크포인트가 가장 좋다.
"깨지기 직전"도 "마지막"도 아닌, 절차가 고른 결과다.

## 4. 최종 평가 — 미사용 seed 2000–2039, std 0.05

| 정책 | 최종 완주 | 개발 완주 | 최근접 중앙값 |
| --- | ---: | ---: | ---: |
| BC (변경 없음) | 33/40 | 13/20 | 3.2 mm |
| M/seed 0 → `step_00061440` | **38/40** | 20/20 | 2.9 mm |
| M/seed 1 → BC 반환 | 33/40 | 13/20 | 3.2 mm |
| M/seed 2 → `step_00099328` | 33/40 | 17/20 | 2.7 mm |

- seed 0의 정책은 BC가 실패한 7개 seed를 성공시키고 2개를 잃어 **+5**다.
- seed 2의 정책은 개발 seed에서 17/20이었지만 최종 seed에서는 BC와 같은 33/40이다
  (BC 실패 4개를 성공, BC 성공 4개를 실패). **개발 성능이 최종 seed로 이어지지 않았다.**
- seed 1은 절차가 BC를 반환했으므로 최종 성능은 정의상 BC와 같다.

## 5. 결론

- **선택 절차는 작동한다.** 세 seed 중 하나는 BC를 그대로 반환했고, 두 개는
  체크포인트를 골랐다. "마지막 체크포인트"나 "깨지기 직전 체크포인트"를 채택하는
  방식이었다면 seed 1에서 BC보다 나쁜 정책을 골랐을 것이다.
- **KL 한계는 갱신 크기를 억제하지만 붕괴를 막지는 못한다.** 0.03에서 seed 0은
  승인 30회, seed 1은 17회에서 무너졌다. 한계 없던 밤 실험(20–30회, 10회)과 같은 범위다.
- **RL의 개선은 학습 seed에 따라 다르다.** 최종 seed 기준으로 3개 중 1개가 BC를
  넘었고(38/40 대 33/40), 1개는 같았고, 1개는 BC 반환이다. 학습 seed 3개로는
  "PPO가 BC를 개선한다"고 말할 수 없고, "어떤 seed에서는 개선이 나오며 절차가
  그것을 고를 수 있다"까지가 이번 근거다.
- 개발 seed와 최종 seed의 불일치(seed 2: 17/20 → 33/40)는 20개 표본의 선택
  잡음을 보여준다. 다음에 선택 표본을 늘리는 것이 후보다.

## 기록

- [selected-policies.json](selected-policies.json) — seed별 반환 정책과 sha256
- `M-seed*-selection.json`, `M-seed*-progress.tsv`, `M-seed*-episodes.jsonl`,
  `M-seed*-manifest.json`, `M-seed*-watch.log`
- [bc-final-eval.json](bc-final-eval.json), `M-seed*-final-eval.json`
- 예비 실행: [P-seed9-progress.tsv](P-seed9-progress.tsv), [P-seed9-train.log](P-seed9-train.log)

```bash
# 예비
scripts/train_approach_rl.py ... --out out/rl-approach/P-seed9 --kl-limit 0.01 \
  --max-actor-updates 10 --seed 9
# 본 실험 (seed 0·1·2)
scripts/train_approach_rl.py --manifest out/rl-approach/candidate.json \
  --out out/rl-approach/M-seed<N> --init-policy out/rl-approach/bc-v1/policy.pt \
  --critic-warmup 40 --policy-lr 3e-5 --critic-lr 3e-4 --kl-limit 0.03 \
  --checkpoint-every 10 --workers 16 --steps 102400 --seed <N> --device cuda:0
scripts/watch_approach_training.py --run out/rl-approach/M-seed<N> \
  --manifest out/rl-approach/candidate.json --baseline out/rl-approach/bc-v1/policy.pt
scripts/select_approach_policy.py --run out/rl-approach/M-seed<N> \
  --baseline out/rl-approach/bc-v1/policy.pt --manifest out/rl-approach/candidate.json \
  --seeds 100-119 --cache out/rl-approach/eval-cache-v2 --out out/rl-approach/M-seed<N>/selection.json
scripts/evaluate_approach_policy.py --policy <selected> --seeds 2000-2039 --sample --std 0.05
```
