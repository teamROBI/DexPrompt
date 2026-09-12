<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

**범위 정정:** 이 계열의 주 실험은 고정 후보의 privileged 팔 접근이다. 다양한 물체의 유형/후보 선정·시범 적응 RL이 실패했다는 결론은 아니다.

# 행동 복제(BC)로 접근을 전이하고, PPO 이어가기의 조건을 찾았다

[PPO 단독 실행](../approach-ppo-v1/README.md)이 성공 경험을 한 번도 얻지 못했으므로,
검증된 접근 시범을 복제해 정책을 초기화했다. 보상·게이트·충돌 제한·완주 판정은
고정했고, 게이트 완화는 하지 않았다.

## 1. 시범: 실제 `env.step()`으로 완주한 실행만 사용

`scripts/collect_approach_demos.py`가 실행기의 웨이포인트 접근을 **정책이 내는
7차원 rate 행동으로 변환**해 환경의 `step`으로 재생하고, 인계에 도달해
`completed`로 끝난 에피소드만 남긴다.

한 궤적만 복제하면 그 궤적 위의 행동만 배우고 되돌아오는 법은 배우지 못하므로,
행동에 잡음을 주입한 재생도 함께 모았다. 잡음으로 경로에서 벗어난 상태의 정답은
"그 상태에서 웨이포인트가 요구하는 행동"이다. 웨이포인트는 매 tick 현재 상태에서
IK를 다시 푸는 구조라 이 복구 라벨이 자연스럽게 나온다.

| | 값 |
| --- | --- |
| 시범 | 깨끗한 재생 1회 + 잡음 0.05 재생 8회 |
| 결과 | **9회 모두 완주** (인계 tick 442–455), 폐기 0 |
| 쌍 | 4,030 (관측 42차원, 행동 7차원) |
| 가장 큰 행동 | rate 제한의 **0.269** |

## 2. 복제

`scripts/train_approach_bc.py`가 PPO와 같은 신경망(512×2 tanh)을 그 쌍에 맞춘다.
200 epoch, Adam 1e-3, 10% holdout.

| | 값 |
| --- | ---: |
| train MSE | 0.000272 |
| holdout MSE | 0.000285 |
| 행동 오차 평균 / 최대 | 0.0117 / 0.1719 |

**행동 오차가 작다는 것은 통과 기준이 아니다.** 판정은 아래의 폐루프 실행이다.

## 3. 폐루프 평가 — home에서 시작, 정책이 만든 상태를 다시 관측

| 실행 | 결과 | 최근접 | 인계 |
| --- | --- | ---: | --- |
| 평균 행동 3회 | **3/3 완주** | 3.2 mm | tick 447 |
| 샘플 행동 10회, std 0.05 | **9/10 완주** | 3.3 mm(중앙값) | — |
| 샘플 행동 10회, std 0.37 (PPO 기본값) | **0/10** | 55.7 mm(중앙값) | — |
| (참조) 기록된 접근 | 완주 | 2.65 mm | tick 451 |
| (참조) PPO 단독 200만 스텝 | 0/N | 18.2 mm(자세 130°) | 없음 |

평균 행동의 최근접 시점(tick 446)에서 위치 3.45 mm, **자세 0.48°**, 물체 이동
0.010 mm다. PPO 단독이 자세 130°였던 것과 대비된다.

**탐색 잡음 크기는 별도의 결정이고, 측정했다.** std 0.37(PPO 기본값)로 샘플링하면
복제한 접근이 완전히 무너진다(밀림 5, 침투 2, 시간 초과 3). 시범의 최대 행동이
0.269이므로 그보다 큰 잡음은 배운 행동을 덮어쓴다. 이어지는 PPO의 초기 탐색을
**0.05**로 정하고 체크포인트·manifest에 기록한다.

## 4. PPO 이어가기: critic 준비 없이 이으면 복제가 파괴된다

BC 체크포인트에서 바로 PPO를 이어 붙였더니 **61,440 스텝(약 60 update) 만에
복제한 행동이 사라졌다.** 그 시점 체크포인트를 평균 행동으로 평가하면 14 tick
만에 물체를 밀어 `quality_approach_displacement`로 끝난다(최근접 227 mm).

원인은 탐색 잡음 증가가 아니다. std는 0.050 → 0.051로 거의 그대로였다.
BC는 actor만 학습하므로 **critic이 초기화 상태**이고, 그 가치 추정으로 만든
advantage(게다가 정규화된)로 actor를 곧바로 갱신한 것이 원인으로 보인다.

그래서 `--critic-warmup N`을 추가했다. 처음 N update는 손실에 **가치 항만** 넣어
actor와 log_std에는 기울기가 가지 않는다.

**정정.** 이 문서의 초판은 warm-up 40으로 다시 이어 붙인 뒤 15,360스텝에서
"복제한 행동이 유지된 채로 학습이 이어진다"고 썼다. 틀렸다. 한 update는
64 × 16 = 1,024스텝이므로 warm-up 40은 **40,960스텝까지 actor와 탐색 잡음이
고정**된다. 그 구간의 완주는 BC 정책을 그대로 실행한 결과이며 warm-up의 효과나
PPO 갱신 후의 보존을 보여주지 않는다.

그 실행(`ppo-bc-v2`)을 583,680스텝까지 진행한 결과는 **붕괴**다. 누적 완주율
1.26%, 최근 구간 완주율 0. 578,560스텝 체크포인트를 평균 행동으로 평가하면
**tick 1의 304.1 mm가 최근접**이다. 즉 목표 쪽으로 아예 움직이지 않는다
(BC 원본은 같은 조건에서 447 tick에 완주). 탐색 std는 0.050 → 0.066으로 커졌다.
이 실행은 보존·평가 후 종료했고, 기록은 `out/rl-approach/archive-ppo-bc-v2/`에 있다.

처리량은 성공이 늘수록 **떨어진다**: 인계 이후 실행기의 닫기·리프트가 한 env
스텝 안에서 최대 950 tick 더 돌기 때문이다(894 → 87 steps/s). 200만 스텝은 이
속도로 여러 시간이 걸리므로, 비교는 체크포인트 단위로 하는 편이 낫다.

## 5. 이것이 보이는 것과 보이지 않는 것

- **시범 전이는 작동한다.** 검증된 실행 하나(+잡음 재생)에서 복제한 정책이
  home에서 시작해 파지까지 완주한다. 이는 RL의 성과가 아니라 **시범 전이의
  성과**다.
- PPO 단독이 실패한 것이 환경·보상의 오류 때문이 아니라는 점도 다시 확인된다.
  같은 환경·같은 보상·같은 게이트에서 복제 정책은 완주한다.
- **RL의 기여는 아직 측정되지 않았다.** BC 단독 대비 PPO 이어가기가 초기 상태
  변화나 접근 실패를 얼마나 줄이는지는 `ppo-bc-v2`가 끝난 뒤 같은 조건으로
  비교해야 한다.
- 이번 평가는 후보 1개·물체 1개·yaw 1개다.

## 기록

- [demonstrations.json](demonstrations.json) — 시범 provenance(후보 sha, 인계 tick, 폐기 목록)
- [bc-manifest.json](bc-manifest.json), [history.json](history.json) — 복제 설정과 손실
- [evaluation-mean.json](evaluation-mean.json),
  [evaluation-sampled-0.05.json](evaluation-sampled-0.05.json),
  [evaluation-sampled-0.37.json](evaluation-sampled-0.37.json) — 폐루프 평가
- [trace-mean.json](trace-mean.json) — tick별 위치·자세 오차
- [ppo-nowarmup-progress.tsv](ppo-nowarmup-progress.tsv),
  [ppo-nowarmup-manifest.json](ppo-nowarmup-manifest.json) — 파괴된 이어가기의 기록

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/collect_approach_demos.py --manifest out/rl-approach/candidate.json \
  --out out/rl-approach/demos.npz --noisy 8 --noise 0.05
OPENBLAS_NUM_THREADS=4 third_party/dexonomy/.venv/bin/python \
  scripts/train_approach_bc.py --demos out/rl-approach/demos.npz --out out/rl-approach/bc-v1
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/evaluate_approach_policy.py --policy out/rl-approach/bc-v1/policy.pt \
  --manifest out/rl-approach/candidate.json --episodes 3 --trace out/rl-approach/bc-v1/trace-mean.json
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/train_approach_rl.py --manifest out/rl-approach/candidate.json \
  --out out/rl-approach/ppo-bc-v2 --init-policy out/rl-approach/bc-v1/policy.pt \
  --critic-warmup 40 --workers 16 --steps 2000000 --seed 0 --device cuda:0
```

## 이어지는 실행

밤 실험([overnight-bc-ppo](../overnight-bc-ppo/README.md))에서 BC를 보존하는 PPO
설정을 찾았다. 낮은 정책 학습률(3e-5)과 시범 유지 항(β=0.1) 모두 붕괴를 막지
못했고, 붕괴 직전 체크포인트 하나(A/seed 0, actor 20 update)만 BC보다 좋았으나
학습 seed 1에서 재현되지 않았다.
