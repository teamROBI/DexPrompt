# 병·캔 시각 student 학습 직전 준비

세 작업을 구현하고 학습 없이 검증했다. **학습과 학습된 시각 정책의 폐루프 성공은 아직
수행하지 않았다.** 아래 성공 대조군은 기록된 teacher 명령을 재생한 결과다.

## 1. 목표 참조 품질

`configs/hx5_student.bottle-can.local.json`에 `goal_reference_policy`를 고정했다.

- native 요청 mask 최소 64픽셀, bbox의 가로·세로 각각 최소 4픽셀.
- 화면 가장자리 2픽셀 띠에 포함된 물체 mask 비율은 최대 10%.
- part 요청은 요청 부위 mask 크기를 검사한다. 경계 비율은 물체 mask로 계산한다.
- 각 카메라의 첫 적격 관측만 저장한다. 현재/과거 관측만 사용하고 현재 mask를 기억으로
  채우지 않는다. 새 요청과 reset에서 초기화하며, 양쪽 참조가 없으면 명령을 거부한다.
- 같은 검사 함수를 학습 데이터와 실시간 GoalMemory가 사용한다. config에 정책이 없으면
  기존 1픽셀 기준을 유지한다. checkpoint config와 코드 해시로 추론 조건을 묶는다.

경계에 닿는 mask를 전부 거부한 초기 기준은 네 궤적 모두 손목 참조를 만들지 못했다.
이를 보존한 `strict-border-rejected-references.json`과 구분해 최종 경계 비율 기준을
사용한다. **부분 가림/잘림을 모두 제거하거나 파지 성능 향상을 입증하는 규칙은 아니다.**

| 시범 | 머리 참조 tick / 픽셀 | 손목 참조 tick / 픽셀 |
| --- | --- | --- |
| 병 `source0_teacher_0002_yaw_000` | 0 / 861 | 345 / 920 |
| 병 `source0_teacher_0006_yaw_000` | 0 / 861 | 451 / 2,943 |
| 병 `source0_teacher_0016_yaw_000` | 0 / 861 | 457 / 2,060 |
| 캔 `source1_teacher_0008_yaw_000` | 0 / 1,347 | 395 / 1,637 |

캔의 기존 손목 참조는 tick 44의 3픽셀이었다. 새 기준은 미래 관측을 앞당겨 쓰지 않고
tick 395에 도달한 뒤에만 해당 참조를 제공한다. 참조 생성 직전·생성 시점·직후를 포함한
**24개 실제 관측**에서 온라인/오프라인 텐서가 정확히 일치했다. 네 궤적의 초기 prefix
제외는 0이다. 미래 이미지 변경이 앞선 학습 입력을 바꾸지 않는 회귀 테스트도 통과했다.

## 2. 통합 데이터

- 데이터: `out/student-bottle-can-v1`.
- cache: `out/student-bottle-can-v1-cache` (2,734,358,032 bytes).
- **병 3개 + 캔 1개 전체 시범, 총 4,170프레임**, 모두 train. 프레임은 독립 시범이 아니다.
- object 조건만 있다. 부위 지시, 새 배치, 새 물체 일반화 데이터로 표현하지 않는다.
- 원본 NPZ 해시는 그대로이고 ID만 입력별 prefix로 구분한다. 품질 판정도 보존했다.
- joint 순서·명령 표현·control_dt·카메라 profile 일치를 검사했다.
- 중복 제거는 원본 궤적 해시와 object/condition/teacher 정체성을 함께 사용한다.
  배열만 같은 다른 물체를 잘못 제거하는 경우를 회귀 테스트에서 발견해 수정했다.
- 실행 모델·초기 상태·teacher 소스와 시각 재렌더 소스를 별도로 복사하고 해시를 확인했다.
  시범당 19개 로컬 archive 파일 검증. 재병합 때 원본 checkout 없이 로컬 archive를
  사용할 수 있으며, 소스 변조·충돌은 거부한다.
- 일반 preflight는 **독립 validation 없음**으로 종료 코드 2. 명시적인
  `--allow-train-only-diagnostic`에서는 종료 코드 0. 독립 평가 데이터가 생긴 것은 아니다.

새 명령 `scripts/prepare_student_diagnostic.py`가 이 진단 전용 병합을 담당한다.
원본 데이터·기존 cache·teacher 실험 결과는 수정하지 않았다.

## 3. 전체 실행 평가와 발견한 오류

`scripts/evaluate_student_suite.py`가 지정한 checkpoint로 각 시범의 archived home 상태를
복원하고 기존 `replay_hx5_student.py`를 실행한다. 팔과 손 **27관절 전체를 student가
명령**한다. TeacherExecutor로 닫기·리프트를 대신 수행하지 않는다. 정책에는 카메라,
목표 mask/기억, 측정 관절만 제공한다. 접촉과 물체 상태는 평가에만 사용한다.

평가기는 지속 리프트와 teacher 접촉 준수 리프트를 구분하고, 힘·침투·최대/종료 접촉군·
전 군 접촉 tick·실패 사유를 기록한다. 전체 시도와 물체별 성공률·물체 평균 성공률을
집계한다. 평가 프로세스 오류를 로봇 실패로 세지 않는다. 기본적으로 training overlap을
거부하고, `--train-task-diagnostic`에서만 이를 명시적으로 허용한다. 반복 seed는 diffusion
샘플링을 바꾸며 물체/배치를 바꾸지 않는다. 코드 불일치 검사를 자동으로 우회하지 않는다.

**기존 평가 오류:** XY 이동 제한을 전체 리프트 구간에 적용해 성공 시범 2개를 접근
밀림으로 실패 처리했다. `controls-old.json`에 실패 결과를 보존했다. 수정된 평가 계약
`contact_acquisition_guard_v2`는 필수 접촉군이 최소 힘을 넘고 연속 0.5초 성립한 뒤에는
물체를 들고 이동하는 것을 접근 밀림으로 처리하지 않는다. 이는 평가에서만 사용하는
측정 기반 구분이다. Teacher phase나 목표 자세를 정책에 넣지 않는다.

침투·힘·테이블/방해물 충돌 제한은 전체 구간에 적용한다. 최종 접촉 유지 리프트 기준도
그대로다. **평가 계약의 변경이므로 이전 whole-rollout XY 평가와 직접 비교하지 않는다.**
물체를 잡기 전 밀림과 연속 접촉 중단, 잘못된 부위 접촉, 획득 후 침투 제한은 회귀 검사했다.

| 대조군 | 접촉 준수 리프트 |
| --- | ---: |
| 기록된 성공 명령 | **4/4** |
| 초기 actuator 명령 유지 | **0/4** |

대조군은 두 조건 모두 각 궤적의 원래 전체 길이로 평가했다. 학습된 정책의 성공률이 아니다.
무학습 전체 모델 검사는 실제 동결 DINOv2 B/L과 전체 diffusion 모델을 사용했다.
병의 두 mask가 모두 비어 있는 프레임과 캔 마지막 프레임에서 손실 **1.242589**, 행동
**[2,64,27]**이 모두 유한했다. 모델 변경·backward·optimizer step·다운로드는 0회.

## 사용자가 실행할 명령

작은 진단 설정은 **batch 2, 3 epochs, warmup 100 steps**다. 수렴이나 성공을 보장하는
설정이 아니다. 현재 저장 방식은 epoch마다 전체 checkpoint를 저장하므로 기존 125 epochs
기본값을 이 진단에 사용하지 않는다. GPU는 학습 시작 시 사용 가능한 장치를 선택한다.

다음 학습 명령은 준비만 했고 **실행하지 않았다**. 저장 경로는 비어 있어야 한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-student/bin/python \
  scripts/train_hx5_student.py train \
  --config configs/hx5_student.bottle-can.local.json \
  --data out/student-bottle-can-v1 --cache out/student-bottle-can-v1-cache \
  --out out/student-bottle-can-bc-v1 --device cuda:0 \
  --allow-train-only-diagnostic
```

학습 후 마지막 epoch를 **미리 고정한 진단 checkpoint**로 평가한다. 저장 번호는 0부터다.
중간 checkpoint를 평가해 보고 좋은 것만 최종 결과로 보고하지 않는다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv-student/bin/python \
  scripts/evaluate_student_suite.py \
  --data out/student-bottle-can-v1 --checkpoint out/student-bottle-can-bc-v1/epoch_0002.pt \
  --out out/student-bottle-can-bc-v1-eval --device cuda:0 \
  --train-task-diagnostic --seed 1000 --repeats 3
```

`summary.json`과 에피소드별 `evaluation.json`·trace NPZ를 확인한다. Loss 감소만으로 P5를
완료 처리하지 않고 home→접근→닫기→접촉 준수 리프트의 실제 성공을 확인해야 한다.

학습 없는 검사 재실행은 새 출력 디렉터리를 사용한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa \
  third_party/dexonomy/.venv/bin/python scripts/evaluate_student_suite.py \
  --data out/student-bottle-can-v1 --out out/NEW_student_controls \
  --check-controls --train-task-diagnostic

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-student/bin/python \
  scripts/train_hx5_student.py dry-run --full-model --device cpu \
  --config configs/hx5_student.bottle-can.local.json \
  --data out/student-bottle-can-v1 --cache out/student-bottle-can-v1-cache \
  --sample-index 1240 --sample-index 4169 --out out/NEW_student_full_check
```

## 검증 기록

관련 테스트 **118 passed**, 변경 Python 파일 Ruff 및 `git diff --check` 통과.
회귀 범위: 참조의 인과성·품질·부위 조건, 중복/ID 충돌/분할, 코드 archive, 평가 겹침
명시·대조군 실패 exit code, 접촉 획득 구분, 기존 student/cache/inference/재생.
전체 저장소의 모든 코드가 오류 없다는 의미는 아니다.

- `bundle.json`, `bundle-integrity.json`: 구성과 원본/로컬 archive 보존.
- `reference-transitions.json`: 실제 참조 전환 24개 관측 대조.
- `verification.json`, `full-model.json`: 최종 참조 기준의 실제 DINO 검사.
- `diagnostic-preflight.json`, `general-preflight.json`: 진단 허용과 독립 평가 부족의 구분.
- `controls-old.json`, `controls.json`: 평가 오류 재현과 수정 후 양성/음성 대조.
- `tests.log`, `ruff.log`, `implementation.json`: 코드 검사와 해시.

P3 물체 확대는 별도 진행 중이다. 이번 완료 범위는 **사용자가 시각 student 학습을
시작하고 전체 실행 결과를 평가할 준비**이며, P5의 학습 정책 완주나 P7 일반화가 아니다.
