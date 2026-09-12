# DexPrompt

Dexonomy의 파지 지식으로 시범을 자동 생성하고, 그 시범으로 지정 물체를 잡는 시각 행동 모델을 학습하는 프로젝트입니다.

1. **시범 생성:** 정확한 시뮬레이터 형상·장면·로봇 상태와 Dexonomy 파지 지식 → 학습 기반 선정·실행 → 접근·닫기·리프트·유지 검증.
2. **시각 행동:** 카메라·물체 선택 Mask/Point/Language·관절 상태 → BC로 학습한 팔·손 연속 행동.

부위 파지는 필수 범위가 아닙니다. 학습된 후보 선정·편집 RL의 이득과 미지 형상 일반화는 별도 검증해야 합니다.

## 지금 읽을 문서

| 목적 | 문서 |
| --- | --- |
| 목표·입력·주장의 범위 | [RESEARCH_PLAN.md](RESEARCH_PLAN.md) |
| 검증 상태·완료 조건 | [PAPER_TODO.md](PAPER_TODO.md) |
| 실행 상태·재개 위치 | [RESUME.md](RESUME.md) |
| 현재 작업 | [현재 작업](plans/CURRENT_WORK.md) |
| 코드 위치·이전 명령 호환 | [폴더 구조](docs/layout.md) |
| 로컬 설치 / lecun | [설치](docs/setup.md) · [서버](docs/lecun.md) |
| 데이터 / 시각 모델 / 목표 계약 | [teacher 데이터](docs/teacher-data.md) · [student](docs/student.md) · [목표 입력](docs/target-contract.md) |
| 학습 전 검사 | [학습 준비](docs/training.md) |
| 과거 실험·계획 | [records](records/README.md) |
| 원고 | [paper](paper/README.md) |

## 현재 근거

- 제한된 물체에서 Dexonomy 후보의 전체 실행·해시 동일 재현을 확보했습니다.
- 최근 배치 실험은 3물체·6후보에서 49/72 성공, 기존 HANDOFF 대비 신규 44건입니다.
  이는 teacher 결과이며 학습된 선정기의 기여나 student 성공률이 아닙니다.
- 기존 A12 시각 자료와 후속 인수 기록은 별도 보관합니다. B 물리의 Milk와 clutter 조건은 분리합니다.
- 1단계 DP3 최소 재현은 학습한 3조건에서 5/9, 확대 실험은 학습조건 3/18·미학습 배치 0/51입니다. [최신 결과](records/experiments/2026-09-13/stage1-dp3-placement/README.md). 목표가 주어진 실행 모델이며 목표 선정이나 별도 시각 VLA 성공은 아닙니다.
- 연구 목표, 진행 상태, 원시 결과를 각 문서에서 분리합니다. 가정 성공 원고를 실측 근거로 사용하지 않습니다.

## 실행과 검사

```bash
# 현재 분류된 스크립트 목록
third_party/dexonomy/.venv/bin/python scripts/run.py --list

# 옛 스크립트 이름도 공통 진입점으로 실행 가능
third_party/dexonomy/.venv/bin/python scripts/run.py generate_grasps.py --help

# 검사 (그래픽 검사는 명시한 MuJoCo backend 사용)
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 third_party/dexonomy/.venv/bin/python -m pytest
```

학습은 lecun에서, 긴 작업은 tmux에서 수행합니다. 재개 전에 RESUME와 실행 상태를 확인하고 중복 실행하지 않습니다.
새 코드의 주 위치는 `src/icra`, 명령은 `scripts/<기능>`, 검사는 `tests/<기능>`, 실험 결과는 `records/experiments`입니다.
기존 실험은 종료됐으며 원격의 고정 실행 사본은 보존합니다. 새 코드 전송은 이후 실행별로 따로 합니다.
