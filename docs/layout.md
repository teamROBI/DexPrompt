# 코드·문서·실험 기록 위치

## 역할별 구조

```text
README.md                시작 안내
RESEARCH_PLAN.md          연구 목표와 범위
PAPER_TODO.md             논문 완료 조건
RESUME.md                 최근 실행 상태·재개 기준
plans/CURRENT_WORK.md     현재 작업만 유지
docs/                    설치·데이터·학습 계약
scripts/
  assets/                메시·URDF·자산 준비
  teacher/               후보 생성·정련·실행·수집
  data/                  시범 인수·묶음 구성
  student/               시각 정책 학습·평가 명령
  rl/                    접근 BC/RL 명령
  stage1_bc/             1단계 MLP BC 실험 패키지
  stage1_dp3/            1단계 DP3 실험 패키지
  experiments/           특정 비교 실험 runner
  diagnostics/           감사·진단·집계
  ops/                   서버·전송·감시·셸 실행
  dexonomy_*/            실행 범위에서만 적용하는 upstream hook
  run.py                 CLI 이름 검색·호환 실행
  _bootstrap.py          직접 실행·기존 스크립트 import 지원
  legacy_paths.json      옛 명령과 새 파일의 대응
src/icra/
  robots/               arm, hand
  sim/                  scene, simulation, objects, placement, object_visuals
  teacher/              execution, contact_control, grasp_frames, probe, stages
  data/                 dataset, recording, outcomes
  vision/               conditioning, visual_observation, goal_memory
  learning/             student*, approach_env, grasp_env, ppo
  integrations/         dexlearn_shim
  paths.py              저장소·자산 경로, 기록 경로 호환, 코드 snapshot
  _compat.py            기존 flat-module import 호환
  __init__.py           가벼운 패키지 초기화
tests/                   teacher/student/rl/simulation/experiments/ops
records/
  experiments/날짜/실험/  성공·실패 결과와 원시 집계
  plans/                 종료된 계획·인계·진행 문서
  scratchpad/            과거 탐색·백업·로그
  maintenance/           구조 변경 원본·경로 대응·검증
paper/                   원고·그림 (가정 성공 초안은 별도 표시)
out/                     생성물과 /data1 자료에 대한 참조
```

공용 라이브러리 이름 `icra`는 유지합니다. `stage1_bc`와 `stage1_dp3`는 `python -m` 실행과
고정 코드 묶음을 쓰는 독립 실험 패키지이므로 내부 구조를 유지합니다.

## 실행과 import

```bash
third_party/dexonomy/.venv/bin/python scripts/teacher/collect_teacher_dataset.py --help
third_party/dexonomy/.venv/bin/python scripts/run.py collect_teacher_dataset.py --help
third_party/dexonomy/.venv/bin/python scripts/run.py --list
```

기존 `python scripts/old_name.py ...`는 `python scripts/run.py old_name.py ...`로 실행합니다.
최상위에 옛 파일의 링크를 다시 나열하지 않습니다. 셸 스크립트는 새 위치로 실행합니다.

새 코드에서는 `from icra.teacher.execution import TeacherExecutor`처럼 역할별 경로를 권장합니다.
기존 `from icra.execution import TeacherExecutor`도 같은 모듈 객체를 가리킵니다.
호환 계층은 필요한 모듈만 불러오며 `import icra`만으로 torch나 MuJoCo를 불러오지 않습니다.

## 기록과 원격 코드

`results`와 `scratchpad`는 `records` 아래로 연결되는 호환 링크입니다. 자료를 복제하지 않습니다.
`out`의 경로는 기존 manifest·재현 기록이 참조하므로 유지합니다. 원시 결과나 모델을
사용 여부만으로 삭제하지 않습니다. 종료된 `.md` 계획은 `records/plans`에 보관합니다.

2026-09-12의 실행 중 파일 고정은 작업 종료 후 해제됐습니다. 그 당시 공지와 해시 목록은
역사 기록이며 현재 파일 위치의 검증 목록이 아닙니다.
과거 manifest의 경로 키와 해시는 수정하지 않습니다. 기록을 읽는 코드는 이전/현재 경로
키를 구분해 읽습니다. 코드 변경 검사를 자동으로 우회하거나 이전 해시를 새 해시로 바꾸지 않습니다.

lecun의 기존 코드 snapshot·모델·데이터는 변경하지 않았습니다. 실시간 동기화도 하지 않습니다.
새 실행 묶음에는 `src/icra` 전체를 복사해야 합니다. `paths.snapshot_library()`가 제공되며
7개 평면 모듈만 골라 복사하는 이전 방식은 새 구조에서 사용할 수 없습니다.
CLI를 복사할 때는 `scripts` 하위 폴더와 `_bootstrap.py`, `legacy_paths.json`을 함께 넣습니다.
과거 실행을 엄격히 재현할 때는 당시 고정 snapshot을 사용합니다.
`ops`의 x0/low-noise/기존 HOPE 검사·비교 감시처럼 특정 snapshot에 묶인 실행기는
그 사본 내부의 옛 경로를 유지합니다. 새 구조의 일반 실행기로 사용하지 않습니다.

## 검증과 복원

[2차 정리 기록](../records/maintenance/2026-09-13-layout/README.md)에 이동 전 snapshot,
`moves.json`, 문서 이동, 테스트·명령 검사를 남깁니다.
[1차 기록](../records/maintenance/2026-09-12-layout/)도 보존합니다.
복원은 해당 snapshot과 경로 대응을 기준으로 필요한 파일만 처리합니다.
작업 전부터 있던 Git 변경이나 다른 작업자의 코드를 전체 초기화하지 않습니다.
