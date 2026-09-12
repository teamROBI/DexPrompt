<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 다음 비교: 같은 물체·다섯 접촉 조건에서 후보 변경

2026-09-08. 당분간 고정 후보에 대한 제어 튜닝을 멈추고 후보 의존성을 확인한다.
실행한 측정과 앞으로 실행할 비교를 구분한다.

## 게이트 진단 재확인

`out/teacher-grip-gate-v1`의 close 600구간을 다시 집계했다.
각 구간의 physics substep 최소 힘을 사용하고 모든 required group의 접촉도 요구한다.

| 각 group의 힘이 넘어야 하는 값 | 동시 통과 구간 수 | 최장 연속 구간 수 |
| --- | ---: | ---: |
| 0.1 N | 146 | 35 |
| 0.5 N | 17 | 15 |
| 1.0 N | 0 | 0 |
| 1.5 N | 0 | 0 |

가장 약한 group의 힘은 최대 0.894758 N이다. 중앙값은 양수인 구간만 포함하면
0.159829 N, 전체 600구간에서는 0이다. 집계 구간을 반드시 명시한다.
리프트 시작에는 25구간 연속 통과가 필요하므로 총 횟수와 연속 유지 시간을 구분한다.

이 결과는 **현재 이식 모델·접근·제어에서 이 후보가 충분한 동시 접촉을 확보하지
못했다**는 근거다. 기하적으로 다섯 접촉이 불가능하거나 어떤 제어로도 불가능하다는
증명은 아니다. Teacher squeeze를 넘긴 명령도 실제 관절 범위나 토크 여유를
소진했다는 증거와는 구분한다.

## 앞선 해석에서 정정한 것

관절 여유를 직접 측정했다. `out/teacher-closure-relief-v2`의 리프트 마지막
구간에서 스무 개 손가락 관절 중 **한계에 닿은 것은 없다**. 여유가 가장 적은
관절이 0.132 rad(약 7.5도)이고, 접촉을 잃은 middle/ring/pinky의 관절도 각각
0.132~0.923 rad가 남아 있었다.

따라서 "더 접을 데가 없어서 회복하지 못했다"는 앞선 서술은 틀렸다. 정확히는
**teacher의 squeeze 목표를 지나 있었을 뿐 기구적으로는 더 접을 수 있었다**.
alpha가 1을 넘은 것은 teacher 목표 대비 비율이며 관절 범위 소진과 다르다.

접촉 손실의 원인을 시간 순서만으로 "힘 부족"이라고 단정한 것도 과했다. 손실
직전의 힘이 0.3~0.5 N이었던 것은 사실이지만, 손실 이후 남은 손가락으로 하중이
옮겨가는 과정을 함께 보아야 인과를 말할 수 있다. 이번 비교에서는 두 국면을
나누어 기록한다.

## 게이트 설정

`ExecutionConfig.lift_force`의 기본값은 `null`이다. 추가 게이트만 끄며 기존의
required contact, 0.1 N, part, 25연속 구간 조건은 유지한다. `1.5`를 지정하면
강한 시작 게이트를 재현한다. 데이터 합격 기준은 바꾸지 않는다.
설정은 수집 provenance에 저장된다.

변경 후 전체 Torch 테스트 224개가 통과했고 수정 파일 Ruff와 diff 검사가 통과했다.

## 준비한 후보와 순서

기존 export를 정적으로 다시 선택해 24후보를 얻었다. Table clearance에서 302건,
open-hand fit에서 33건을 제외했다. 이는 동적 실행 성공률이 아니다.
전체 목록은 `out/teacher-candidate-screen-v2.json`이다.

첫 비교는 같은 bottle·`11_Power_Sphere`·동일한 다섯 required group의 네 자세다.
기존 `0_14_grasp.npy`와 새로운 `0_15`, `0_16`, `0_18`을
`candidates.json`에 원본 SHA·열기·squeeze·접촉 조건과 함께 고정했다.
Yaw는 0이다. 다른 물체·다른 type·두 손가락 pinch는 별도 비교군으로 둔다.

1. 제어기, 모델, 질량, 마찰, 20 ms 주기, scale을 고정한다.
2. 네 후보 각각에 `baseline.json`과 `strict.json`을 적용한다.
   두 설정은 리프트 시작의 추가 게이트만 다르며 유지 목표는 모두 4 N이다.
3. 후보마다 동적 home approach를 검사한다. 원래 후보의 +X 경로를 다른
   자세에 무조건 강제하지 않는다. 대응하는 게이트 비교에는 같은 초기 상태와
   접근 경로를 사용한다.
4. 접근 실패와 접촉 형성 실패를 분리한다. 접근하지 못한 후보를 안정 접촉
   불가능으로 분류하지 않는다. 후보에 맞춰 required group을 임의로 줄이지 않는다.
5. 첫 진단은 가능하면 headless로 한다. 기존 `closure-relief/probe.py`는
   원래 후보 전용이므로 다른 후보에 그대로 재사용하면 안 된다.
   아래는 기존 정식 collector로 전체 경로와 영상을 기록하는 실행 명령이다.

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/collect_teacher_dataset.py collect \
  --manifest results/2026-09-08/candidate-screen/candidates.json \
  --execution-config results/2026-09-08/candidate-screen/baseline.json \
  --out out/candidate_compare_baseline --max-candidates 4 --max-yaws 1 \
  --width 64 --height 64
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/collect_teacher_dataset.py collect \
  --manifest results/2026-09-08/candidate-screen/candidates.json \
  --execution-config results/2026-09-08/candidate-screen/strict.json \
  --out out/candidate_compare_strict --max-candidates 4 --max-yaws 1 \
  --width 64 --height 64
```

이 여덟 실행은 아직 시작하지 않았다. 학습도 실행하지 않았다.

## 기록할 항목

- Group별 힘 시계열, 각 임계값의 동시 통과 수와 최장 연속 시간.
- 실제 접촉 링크·위치와 원본 required group 정의.
- 손 기준 물체의 상대 이동·회전. 닫기 시작과 리프트 시작을 별도 기준으로 둔다.
  상대 이동에는 구름·자세 변화도 포함되므로 순수한 미끄럼으로 단정하지 않는다.
- 명령과 실제 관절각, 관절 범위까지의 여유, actuator torque 포화, teacher
  squeeze로부터의 편차. 분모가 작은 관절의 배율은 해석에 주의한다.
- 침투, 전체 손가락 압력, 지지면 접촉, lift/hold, 종료 이유, 계산 시간.

게이트 없는 실행은 먼저 lift로 넘어가므로 닫기 구간 길이도 함께 기록한다.
다른 게이트의 궤적에서 실행하지 않은 600구간을 추정해 채우면 안 된다.
모든 임계값을 같은 600구간에서 비교하려면 리프트로 넘어가지 않는 별도의
closure-only 진단이 필요하다.

## 결과에 따른 판단

- **같은 물체·type의 다른 자세가 성공:** 원래 후보를 주 디버깅 기준에서 빼고
  실행 가능한 후보를 고르는 절차를 발전시킨다.
- **전부 실패:** teacher 자체의 문제로 단정하지 않는다. 원래 Dexonomy
  evaluator와 이식 후 geometry/scale/root frame, 물리 설정, 접근을 비교한다.
  원래 환경에서 성공이 재현되는 후보를 대조군으로 삼는다.
- **접촉은 형성되지만 lift 중 손실:** 상대 이동, 접촉 배치, 하중 이동을 조사한다.
  힘을 일괄적으로 올리고 내리는 조정만 반복하지 않는다.
- **특정 제어에서만 성공:** 제어 의존성이 있으므로 후보 선택 문제로만 귀결하지 않는다.

1.5 N은 진단용 가설이며 안정 파지에 보편적으로 필요한 힘이 아니다.
이 비교가 정리되기 전에는 새로운 tactile/VLA/RL 구현을 추가하지 않는다.
