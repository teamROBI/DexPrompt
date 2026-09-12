<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 닫기 강도와 리프트 정지의 결합 수정

2026-09-08. 범위는 `src/icra/contact_control.py`, `src/icra/execution.py`와
관련 회귀 테스트이다. Teacher, student, 학습 데이터 판정 기준은 바꾸지 않았다.
학습은 실행하지 않았다.

## 확인한 원인과 수정

- 기존 기록은 닫기 중 1.99975 mm 침투를 보이지만, 그 뒤 침투가 감소했다가
  리프트 중 다시 증가한다. 초기 최대 침투가 그대로 유지된 것은 아니다.
- 기존 닫기는 모든 접촉이 생기면 리프트 전에도 2 N에서 8 N으로 목표를
  올렸다. 이제 닫기에서는 2 N을 유지하고, 리프트 목표는 기본 4 N이다.
- 필수 접촉점이 사라지면 해당 손가락의 다른 링크가 이미 누르고 있어도
  접촉 탐색이 계속될 수 있었다. 전체 손가락 압력이 과하면 탐색보다
  압력 완화를 우선한다. 접촉을 만드는 도중의 전체 손가락 과압 기준은
  30 N 한계의 0.75배이며, 운반 중에는 목표 힘을 기준으로 확인한다.
- 리프트 중 손가락별 침투가 1.5 mm를 넘으면 닫기를 완화한다. 추가적인
  과압/침투 완화는 실제 관절 명령을 측정 각도 방향으로 최대 0.001 rad씩
  이동하도록 제안하고, 누적된 기하 보정값에도 반영한다. 닫기 진행량만
  낮추고 기하 보정값을 남기는 재가압을 방지한다.
- `guarded()`는 운반 중 팔 이동과 손가락 변화를 분리해서 검사한다.
  손가락 변경 때문에 팔 상승까지 무조건 거부하지 않으며, 정지 명령을
  택하기 전에 압력 완화를 검사한다. 실제로 채택한 팔/손가락 비율을
  각각 반영하여 실행하지 않은 보정은 누적하지 않는다.
- 접촉 손실 시 상승 보류는 유지한다. 그동안 손가락 회복 제어는 계속된다.
  접촉을 처음 만드는 단계에서는 기존 지지를 보존한 뒤 전체 손을 푼다.
  초기 시험에서 조기 전체 완화가 접촉 형성을 불안정하게 했기 때문이다.

2 mm 침투, 30 N 전체 손가락 압력, 10 cm 상승 및 접촉 유지 시간 등의
품질 기준은 그대로다. 위 제어 목표가 품질 통과를 보장하지는 않는다.

## 고정 과제 진단

동일한 bottle/Power-Sphere/yaw 0, 기존 MJB와 초기 상태에서 확인했다.

| 제어 | 최대 interval-minimum 상승 | 최대 전체 손가락 압력 | 종료 |
| --- | ---: | ---: | --- |
| 기존 v4 | 8.4247 cm | 29.1299 N | `no_quality_feasible_command` |
| 수정 제어, 목표 2 N | 7.6906 cm | 19.6844 N | `lift_timeout` |
| 수정 제어, 목표 4 N | 9.8499 cm | 23.3391 N | `lift_timeout` |

수정 제어의 두 행은 `out/teacher-closure-probe-v5`와
`out/teacher-closure-probe-v5-hold4`의 headless 진단이다. 이미지나 학습
데이터를 생성하지 않았다. 2 N/4 N은 같은 제어에서 목표 힘만 비교했다.
기존 v4와의 비교는 제어 구조도 달라서 목표 힘 하나의 인과 비교가 아니다.
기본 4 N은 이 고정 과제에서 선택한 초기값이며 일반 물체나 실기의 최적값이 아니다.

headless 진단의 재실행 예시(각각 새 출력 경로 사용):

```bash
OPENBLAS_NUM_THREADS=1 third_party/dexonomy/.venv/bin/python \
  results/2026-09-08/closure-relief/probe.py --out out/closure_check_4n
OPENBLAS_NUM_THREADS=1 third_party/dexonomy/.venv/bin/python \
  results/2026-09-08/closure-relief/probe.py --out out/closure_check_2n \
  --config results/2026-09-08/closure-relief/hold2.json
```

`probe.py`는 이 작업공간의 고정 v4 아카이브를 사용하는 진단 스냅샷이다.
`trace.npz`에는 물리 신호, `guards.json`에는 미리보기 거절 사유와 승인한
팔/손가락 비율을 기록한다. 현재 소스로 다시 실행하므로 소스 버전도 함께 보존한다.

**10 cm 이상에서 필수 접촉을 유지하는 성공은 아직 없다.** 닫기 단계의
최대 침투도 기존 1.99975 mm와 같다. 단순히 더 약하게 닫으면 해결된다는
결론은 지지되지 않는다. 초기의 일괄 완화 시험은 닫기/리프트 초반에 더
일찍 중단되었고, 그 결과를 성공으로 채택하지 않았다.

최종 실제 수집 `out/teacher-closure-relief-v2`도 위 4 N 진단과 동일하다.
전체 접근과 영상 기록을 포함하여 1,450개 제어 구간(29초)을 실행했다.
침투 최대 1.99975 mm, 압력 최대 23.3391 N, 상승 최대 9.8499 cm,
robot-table/obstacle 침투와 distractor 교란은 0이다. compiled physics profile이
기존 v4와 정확히 같음을 확인했다. `lift_timeout`으로 종료되며 성공 시연은 0개다.

종료 시 필수 접촉은 `[1, 1, 0, 0, 1]`이다. 마지막 100개 구간에서 약지
그룹은 접촉 0회, 중지는 43회였다. 동일 headless 진단의 마지막 100개
미리보기는 모두 팔 승인 비율 1이며, 실제 손가락 명령도 계속 변한다.
따라서 남은 정지는 `fraction=0`에 의한 전체 명령 고정으로 설명되지 않는다.
중지/약지 접촉 회복이 되지 않아 상승 요청이 보류되는 문제가 남는다.

`out/teacher-closure-relief-replay-v2`에서 저장 명령 1,450개를 재실행했다.
관절 상태·시간·접촉력·침투·높이·교란 신호 최대 오차가 모두 0이고,
접촉 비트 불일치도 0이다. 실패 판정도 일치한다. 이미지는 이 수치 재현에서
독립적으로 검증하지 않는다. 작은 증거 사본은 `collection_manifest.json`,
`replay_summary.json`, `probe_2n_summary.json`, `probe_4n_summary.json`에 보존했다.

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/collect_teacher_dataset.py collect \
  --manifest results/2026-09-08/fixed-student/selection.json \
  --out out/closure_full_check --width 64 --height 64 \
  --max-candidates 1 --max-yaws 1 --directions x
```

## 소프트웨어 검증

- Torch/MuJoCo 3.3.3 환경 전체 테스트: 218개 통과.
- 수정된 접촉/실행 영역 회귀 테스트: 51개 통과.
  Torch/MuJoCo 3.3.3과 별도 MuJoCo 3.12 환경에서 모두 확인했다.
- 신규 검사는 리프트 전 목표 힘 증가, 접촉 소실 중 과압 닫기,
  침투 완화, 기하 보정값 재가압, 팔/손가락 독립 승인, 접촉 형성 단계의
  조기 전체 완화를 다룬다.
- 수정 Python 파일 Ruff 통과. 물리 설정과 성공 판정 기준은 완화하지 않았다.

## 실제 HX5-D20 tactile과의 관계

현재 제어는 MuJoCo의 접촉 정보를 사용한다. 실제 촉각 센서 입력이나
그 신호의 교정 모델을 구현한 것은 아니다. HX5-D20에는 손끝마다
3×3 촉각 배열이 있으며, ROBOTIS는 접촉 감지 후 힘을 유지하는 제어
예제를 제공한다. [공식 제어 가이드](https://ai.robotis.com/hands/technical_story_hands_tactile_grasp.html)

공식 사양의 센서 출력은 0–255 값이다. MuJoCo의 N 단위 접촉력 목표를
이 원시 값에 그대로 대입하면 안 된다. 무접촉 기준값, 필터링, 포화 및
필요한 힘 교정을 먼저 확인해야 한다.
[공식 사양](https://ai.robotis.com/hands/hardware_hands.html)

현재 고정 teacher는 각 손가락의 마지막 두 링크 중 하나를 접촉으로 인정한다.
손끝 센서가 감지하는 표면과 같은 정의는 아니다. 실기용 피드백과 시뮬레이터
품질 판정을 구분하고, 센서가 실제 관측할 수 있는 접촉 영역을 먼저 맞춘다.
Dexonomy/VLA의 접근·파지 목표 아래에 촉각 기반 닫기/유지 제어를 두는
구성이 다음 구현 후보이며, 이번 수정에서 VLA 입력이나 실제 센서 드라이버는
변경하지 않았다.
