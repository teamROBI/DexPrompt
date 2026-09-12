<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

**범위 정정:** 이 계열의 주 실험은 고정 후보의 privileged 팔 접근이다. 다양한 물체의 유형/후보 선정·시범 적응 RL이 실패했다는 결론은 아니다.

# 접근 RL 통합 검증 (학습 없음)

고정 후보 하나로 팔 접근만 RL로 학습할 환경을 붙이고, **학습 없이** 통합을
검증했다. PPO는 실행하지 않았고 gradient step도 없다.

## 고정한 후보

| 항목 | 값 |
| --- | --- |
| 예측 후보 / 변형 | `grasp_38` / `grasp_38_r120_d0` |
| 정렬 실험 / seed | `big1_offset_s0_hx5_d20` / 0 |
| 정련 파일 | `…/33_Inferior_Pincer/core_bottle_6444…/floating/scale011/0_2_grasp.npy` |
| SHA256 | `05b63bfb299053e9edf05f61741c849b103d7863c3e462a90fd30f4c73d983a9` |
| 기준 실행 | `executed-offset/teacher_0002_yaw_000` (episode sha `3b3bf3df72eb…`) |
| 물체 / scale / yaw | `core_bottle_6444875e3217bf891f5f48a891d827bd` / 0.11 / 0.0 |
| 템플릿 접촉점 | `a24743aa4cf5…`, contact index 2 |

이 선택은 첫 통합 검증을 위한 것이며 보정 offset의 우월성을 뜻하지 않는다.

## 구조: 접근만 학습, 파지는 기존 제어기

`icra.approach_env.ApproachEnv`가 팔 7관절만 명령하고 손가락은 해당 타입의
calibrated open에 둔다. 인계 조건을 만족하면 `TeacherExecutor.close_and_lift()`로
제어가 넘어가고, **에피소드의 성패는 그 제어기의 판정**이다. 도착 자체는 성공이
아니다.

이를 위해 `execution.py`를 두 번 리팩터링했다. 동작은 바꾸지 않았다.

1. `run()`에서 닫기·리프트 루프를 `close_and_lift()`로 분리했다.
2. 웨이포인트 접근을 `approach_targets()` 제너레이터로 분리하고 `approach()`가
   그것을 소비하게 했다. 같은 동작을 다른 루프(이 환경)에서도 그대로 돌린다.

**검증:** 두 리팩터링 후 고정 후보를 다시 수집했더니 episode sha256이
`3b3bf3df72eb…`로 **리팩터링 이전과 완전히 동일**했다(974 ticks, completed).

## 인계 조건

위치 오차뿐 아니라 자세·충돌·물체 밀림을 함께 본다. 다섯 tick 연속 만족해야
인계한다(스쳐 지나가는 자세를 도착으로 세지 않기 위해서다).

| 조건 | 값 |
| --- | --- |
| 손목 위치 오차 | ≤ 5 mm |
| 자세 오차 | ≤ 0.05 rad (2.9°) |
| 물체 이동(안착 위치 대비) | ≤ 5 mm |
| 로봇–물체 / 테이블 / 방해물 침투 | ≤ 1e-6 m |
| 접촉군 | 하나도 닿지 않았을 것 |
| 연속 유지 | 5 tick |

## 0. 지적받은 두 오류를 고쳤다

**보정 전 관절 목표를 별도 상태로 관리한다.** 이전에는 `r.targets()`(= 실행기가
마지막으로 쓴 actuator 명령, **중력 보정 포함**)를 목표로 읽어 거기에 행동을
더했다. 그러면 보정 항이 tick마다 다시 더해져 영 행동만 반복해도 목표가
스스로 이동한다. 이제 `ApproachEnv.arm_target`이 보정 전 목표를 들고 있고,
중력 보정은 `executor.action()`이 명령을 만들 때 한 번만 적용된다.
`reset`은 측정값으로 이 상태를 초기화하고, `apply`(절대 목표)와 `step`(rate)이
같은 상태를 갱신한다. 회귀 테스트: 중력 보정 0.3인 환경에서 영 행동 20회 후
목표 이동량이 정확히 0, 그리고 두 경로가 같은 목표를 공유하는지.

**접근 보상·진단값은 인계 직전에 확정한다.** 이전에는 `close_and_lift()`가
끝난 뒤 보상과 진단을 읽어, 15 cm 정상 상승이 "접근 중 물체를 밀었다"로
집계되고 shaping도 리프트 후 상태로 계산됐다. 이제 인계 시점에 접근 보상과
`arrival` 진단을 확정하고, 그 뒤에는 실행기의 성공/실패 보상만 더한다.
`_info`의 `position_error`·`displacement`는 접근이 인계한 값이고, 리프트 후
상태는 `final_*`로 따로 적는다. 회귀 테스트: 깨끗한 리프트(물체 15 cm 이동)
후 보상이 `success + time`뿐이고 접근 진단의 이동량이 0인지, 실패 시에도
`failure + time`만 한 번 계산되는지.

**접근 시간 초과는 terminal이다.** 한때 truncation(수집 편의상 끊긴 과정)으로
두었으나, 이번 과제 정의는 "**600 tick 안에 접근하고 인계 후 파지를 완주하기**"이고
관측에도 경과 시간 비율이 들어간다. 그러므로 제한 시간 초과는 계속할 수 있는
과정을 끊은 것이 아니라 **제한 시간 내 실패**다. 처리: `terminated=True`,
`truncated=False`, terminal potential 0, 정의한 실패 보상 적용, bootstrap 없음.
PPO의 64-step rollout 경계는 에피소드 종료로 처리하지 않는다(수집만 이어진다).

## 1. 양성 대조군 — 기존 성공 실행을 7차원 행동으로 재현

기록된 접근 목표를 rate 제한의 역변환(`scripted_actions`)으로 7차원 행동으로
바꿔 **`env.step` 경로로** 재생했다(`apply` 우회 아님).

- **인계 tick 451**, 인계 시점 위치 오차 **2.65 mm**, 자세 오차 **0.15°**,
  물체 이동 **0.01 mm**.
- 인계 후 `close_and_lift()`가 474 tick 만에 **completed**, 최종 성공 판정.
- 필요한 최대 행동 크기는 rate 제한의 **0.184**(한 tick 최대 관절 변화 0.92%,
  허용 5%). 기록된 동작은 정책의 행동 공간 안에 있다.
- 같은 동작을 절대 목표(`apply`) 경로로도 돌렸다: 인계 tick 451, completed로
  동일하다. 닫기 tick 수만 474 대 476으로 다른데, rate 왕복 변환의 부동소수점
  오차 때문이며 bit 단위 동일을 주장하지 않는다.

## 2. 음성 대조군 — 실패가 성공으로 처리되지 않는지

| 대조군 | 결과 | 상태 |
| --- | --- | --- |
| 제자리 유지(영 행동 600회, `step` 경로) | 성공 아님 | `approach_timeout`, **목표 이동량 0.00e+00** |
| 60%에서 중단 | 성공 아님 | 인계 없음(`approaching`) |
| 접근축으로 90° 롤한 자세로 도착 | 성공 아님 | 인계 없음(`approaching`) |
| 접근 중단 후 테이블로 하강 | 성공 아님 | `quality_approach_displacement` |
| 물체를 밀고 들어가기 | 성공 아님 | `quality_penetration` |

"롤한 자세로 도착"은 위치는 맞지만 자세 조건에서 걸려 인계되지 않는다.
"테이블로 하강"은 테이블 침투보다 물체 밀림 한계에 먼저 걸렸다. 관측된 상태를
그대로 기록한다.

## 3. PPO 트레이너 연결 — 학습 없이, 그리고 검사가 판정한다

`scripts/train_approach_rl.py`가 이 환경에 PPO를 붙인다(행동 7차원, worker
프로세스, HORIZON 64, GAMMA 0.99, LAMBDA 0.95). `--check`는 학습 없이 루프를
돌리되 **판정한다**: 아래 조건이 하나라도 어긋나면 `check.json`에 사유를 남기고
**종료 코드 1**로 끝난다.

- 관측 폭이 manifest와 다르거나 행동이 7차원이 아니면 실패
- reward·value·advantage·return·loss 중 하나라도 비유한이면 실패
- 기록된 접근 재생이 인계에 도달해 `completed`로 끝나지 않으면 실패
- 영 행동이 `approach_timeout`·`terminated=True`·`truncated=False`로 끝나지
  않거나 성공으로 집계되면 실패

또한 검사는 advantage에서 멈추지 않고 **학습과 같은 함수**(`ppo_losses`)로
log-probability·ratio·actor/critic loss·entropy까지 계산해 유한성을 확인한다.
그동안 `torch.Tensor.backward`와 `optimiser.step`은 **호출되면 예외를 던지도록
교체**되어 있어 "학습 안 함"이 약속이 아니라 강제다.

검사 결과(worker 4, rollout 3회, 관측 41차원, 행동 7차원):

| 항목 | 결과 |
| --- | --- |
| rollout 형태 | `[64, 4]` × 3 |
| 수집 스텝 | 768 (233 steps/s) |
| reward·value·advantage·return 유한성 | 모두 True |
| loss (policy / value) | 2.85 (-0.0000 / 5.72), 9.90, 5.27 — 모두 유한 |
| 종료/reset | 에피소드 9회 종료 후 자동 reset, 이어서 수집 계속 |
| backward / optimizer.step | 0 / 0 (호출 시 예외) |
| 최종 판정 | **passed**, 종료 코드 0 |

무학습 정책은 곧바로 테이블에 부딪히므로 9회 모두 `quality_table_collision`
(terminated)이었다. 인계와 시간 초과 경로는 같은 프로세스·같은 코드 해시로
따로 확인해 `check.json`에 함께 기록했다:

- 기록된 접근 재생 → **인계 tick 451, completed, 성공**
- 영 행동 → **`approach_timeout`, terminated True / truncated False**

**검사가 실제로 실패하는지도 확인했다.** `--approach-steps 40`으로 돌리면
기록된 접근이 인계에 도달하지 못하고, 검사는 사유 두 건을 적고 **종료 코드 1**을
반환한다. 판정 로직은 회귀 테스트 11개(`tests/test_approach_trainer.py`)로
따로 고정했다.

## 3-1. 학습 시작 기록

`manifest.json`의 `training_started`는 이제 **실제로 gradient가 적용되는 순간**
True로 바뀌고 `training_started_at`이 함께 기록된다(첫 `optimizer.step()`에서
manifest를 다시 쓴다). `--check` 실행에서는 계속 False다.

## 4. 시간 제한 분리

| 구간 | 한계 | 기록된 성공 실행 사용량 |
| --- | ---: | ---: |
| 접근 | 600 tick = 12.0 s | 451 tick |
| 닫기·리프트 (`close+lift+hold`) | 950 tick = 19.0 s | 476 tick |
| 합계 | 1550 tick = 31.0 s | 927 tick = 18.5 s |

기존 단일 기본값 12초는 이 성공 실행의 19.48초보다 짧다. 두 구간을 분리했으므로
닫기·리프트의 정상 소요 시간이 정책 실패로 계산되지 않는다.

## 5. 할인율을 접근 길이에 맞췄다 (gamma 0.99 → 0.999)

보상 구성은 그대로 두고 **시간 척도만** 고쳤다. 성공 보너스 +25는 451번째
행동에서 받는데, 0.99에서는 에피소드 시작 시점 가치가 0.27이다. 제자리에서
모으는 보조 보상보다 작아서 "시도하지 않기"가 유리해질 수 있는 설정이다.
0.999에서 같은 보너스는 15.9의 가치를 갖는다.

환경의 보조 보상(potential)과 PPO·GAE가 **같은 값**을 쓰도록, 할인율을
`icra.approach_env.GAMMA` 한 곳에 두고 트레이너가 그것을 import한다.

학습 없이 세 동작을 실행해 시작 시점 할인 누적 보상을 다시 측정했다.

| 동작 | 결과 | gamma 0.99 | **gamma 0.999** |
| --- | --- | ---: | ---: |
| 기존 성공 접근 | 451 tick 인계 → 완주 | 12.715 | **25.738** |
| 영 행동 유지 | 600 tick 시간 초과 | 12.374 | **−4.811** |
| 무작위 행동 | 테이블 충돌 | 12.208 | **−5.979** |

성공과 대기의 차이가 0.34에서 **30.5**로 벌어졌다. "95%는 451 tick에 성공하고
5%는 일찍 충돌하는" 가상의 정책도 이제 기대 보상 약 24.2로 대기(−4.8)보다
확실히 높다. 이 값이 최적이라는 주장은 아니며, 9초짜리 접근을 평가하기 위한
시작값이다.

함께 고친 것:

- **밀림 벌점의 과거 최대값(`pushed`)을 관측에 넣었다.** 같은 현재 상태가
  과거에 따라 다른 보상을 받는 불일치를 없앤다. 관측은 41 → 42차원이 됐다.
- 에피소드의 **누적·할인 누적 보상**을 진단값으로 기록한다(정책은 보지 않는다).
  보상 합만으로 학습 진전을 판단하지 않기 위해, 인계율·완주율과 함께 본다.
- 거리·자세 potential 구조, 성공 보상, 충돌 벌점, 침투·충돌 제한, 완주 판정은
  **바꾸지 않았다.** 이번에는 할인율만 움직였다.

## 6. 첫 학습 실행

위 검사를 모두 통과한 뒤 tmux 세션 `icra-ppo`에서 학습을 시작했다.

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/train_approach_rl.py \
  --manifest out/rl-approach/candidate.json \
  --out out/rl-approach/ppo-v1 \
  --workers 16 --steps 2000000 --approach-steps 600 --seed 0 --device cuda:0
```

`tmux attach -t icra-ppo`로 붙고, 로그는 `out/rl-approach/ppo-v1.log`,
지표는 `out/rl-approach/ppo-v1/progress.tsv`(steps, reward, episodes,
handover_rate, success_rate, window_success_rate, steps/s), 체크포인트는
`policy.pt`다. `manifest.json`의 `training_started`는 첫 gradient가 적용된
`2026-09-09T00:22:40+0900`에 true로 기록됐다.

초기 속도는 약 577–700 steps/s이고 200만 스텝은 한 시간 안팎이다. 인계가
일어나기 시작하면 한 스텝 안에서 닫기·리프트가 최대 950 tick 더 돌기 때문에
느려진다. 볼 것은 보상 합이 아니라 **인계율과 완주율**, 그리고 충돌이 줄면서
접근하는지 아니면 그냥 정지하는지다.

## 검사와 기록

- 회귀 테스트 11개(`tests/test_approach_env.py`) + 11개(`tests/test_approach_trainer.py`): 인계 조건 네 가지가 모두
  필요한지, 연속 유지 요구, 성패가 실행기 판정에서 오는지, 품질 위반이 접근을
  그 자리에서 끝내는지, 접근 시간 제한, 행동의 rate 제한.
  추가된 것: 중력 보정 상태 누적 없음, 두 명령 경로의 목표 공유, 기록 목표 →
  행동 변환, 리프트를 접근 밀림으로 벌점 처리하지 않음, 시간 초과의 truncation.
  트레이너 쪽: 관측·행동 폭, 유한성 다섯 가지, 양성 대조군 실패 감지, 시간
  초과 판정(terminated/truncated·성공 오집계), 모든 rollout 판정, 그리고
  `ppo_losses`가 학습이 쓰는 그래프를 그대로 만드는지.
- 전체 테스트 **312 passed**, 변경 파일 Ruff 통과.
- [verification.json](verification.json) — 대조군 수치와 설정
- [ppo-check.json](ppo-check.json), [ppo-manifest.json](ppo-manifest.json) —
  트레이너 검사 결과와 설정·코드 해시
- [candidate.json](candidate.json) — 고정 후보와 부모 귀속

재현:

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/verify_approach_env.py --manifest out/rl-approach/candidate.json \
  --out out/rl-approach/verification.json
```

## 아직 하지 않은 것

- 학습. 트레이너는 연결됐지만 gradient step은 한 번도 실행하지 않았다.
  보상 계수 탐색과 학습 곡선도 없다.
- 보상 항(`ApproachWeights`)은 가설이며 측정값이 아니다.
- 기존 접근 실행기는 비교 기준으로 남겨 두었다. 경로 생성기는 바꾸지 않았다.
- 이 검증은 후보 1개·물체 1개·yaw 1개다.

## 이어지는 실행

첫 학습 결과는 [2026-09-09/approach-ppo-v1](../../2026-09-09/approach-ppo-v1/README.md)에
있다. 200만 스텝에서 인계 0회지만 정지한 것은 아니며(최근접 303.9 → 18.2 mm,
에피소드 길이 258 → 600 tick), 왜 마지막 13 mm에서 멈추는지를 보상 분해로
확인했다.
