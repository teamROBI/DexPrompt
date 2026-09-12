# 현재 상태와 재개

2026-09-13. 기존 실험은 종료됐습니다. 구조 정리 전에 turing과 lecun의 프로젝트 tmux·실험 프로세스가 없음을 확인했습니다. 이번 정리에서 새 학습·데이터 수집은 시작하지 않았습니다.

## 최근 결과

- [1단계 데이터 계약](records/experiments/2026-09-12/stage1-bc-data-contract/README.md): 기본 풀 77개, 4물체. HOPE 75개 중 접촉 획득 계약 v2 통과는 72개입니다. 원래 teacher 판정과 구분합니다.
- [DP3 최소 재현](records/experiments/2026-09-12/stage1-dp3-feasibility/README.md): 학습한 3조건에서 5/9 완주.
- [동일 조건 MLP 비교](records/experiments/2026-09-12/stage1-goal-bc/DP3-MATCH.md): 0/9. 모델·행동 묶음·EMA 차이는 분리되지 않았습니다.
- [DP3 배치 확대](records/experiments/2026-09-13/stage1-dp3-placement/README.md): 55시범 학습, 기록 재생 23/23, 학습조건 3/18, 미학습 배치 0/51. 새 형상 일반화나 VLA 성능 결과가 아닙니다.
- [시각 인수 기록](records/experiments/2026-09-12/followup-visual-intake/README.md)과 [teacher 인계](records/experiments/2026-09-12/teacher-followup-handoff/README.md)는 별도 자료입니다. 종료된 작업을 과거 계획만 보고 재시작하지 않습니다.

## 코드 구조와 재현

- [폴더 구조](docs/layout.md). 스크립트는 기능별 하위 폴더, 공용 라이브러리는 `src/icra`의 역할별 패키지에 있습니다.
- 과거 CLI 이름은 `python scripts/run.py old_name.py --help`로 찾습니다. 기존 `icra.execution` 등의 import도 호환됩니다.
- [이번 구조 변경 기록](records/maintenance/2026-09-13-layout/README.md)에 이전 파일 snapshot·경로 대응·검증 결과를 둡니다.
- 과거 데이터·manifest·source hash와 원격의 고정 코드 snapshot은 그대로입니다. 엄격한 과거 실험 재현에는 해당 snapshot을 사용합니다.
- 이번 변경은 turing 작업 트리 정리입니다. lecun에 실시간 동기화하지 않았습니다. 이후 실험에는 새 코드 묶음을 따로 전송·해시 확인하며 기존 실행 사본을 덮어쓰지 않습니다.

[현재 작업](plans/CURRENT_WORK.md), [연구 목표](RESEARCH_PLAN.md), [논문 완료 조건](PAPER_TODO.md)을 함께 읽습니다. 이전 진행 일지는 [records/plans](records/plans/)에 보관했습니다.
