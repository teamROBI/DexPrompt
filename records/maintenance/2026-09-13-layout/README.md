# 구조 정리 2차 — 2026-09-13

기존 실험 종료 후 보류된 코드 이동과 문서 정리를 진행했습니다. 새 학습·후보 수집·원격 동기화는 하지 않았습니다.

## 변경

- Python CLI 31개와 셸 5개를 `scripts` 기능별 폴더로 이동했습니다. 최상위 파일은 `run.py`, `_bootstrap.py`, `legacy_paths.json` 세 개입니다.
- 공용 모듈 27개를 `icra.robots/sim/teacher/data/vision/learning/integrations`로 이동했습니다. 최상위는 `__init__.py`, `paths.py`, `_compat.py`입니다.
- 남아 있던 테스트 4개를 기능별 폴더로 이동했습니다. 최상위는 공통 `conftest.py`입니다.
- 총 67개 코드·테스트 파일 이동. 독립 실행 패키지인 `stage1_bc`, `stage1_dp3`와 Dexonomy hook은 내부 구조를 유지했습니다.
- 종료된 계획 두 개와 이전 CURRENT_WORK·RESUME 내용을 `records/plans`로 보관했습니다. 현재 문서는 실행 종료 상태와 최근 결과를 가리킵니다.
- CLI 호출·저장소 루트 계산·문서 링크·소스 해시 조회를 새 위치에 맞췄습니다. 이전 Python import는 새 모듈과 같은 객체를 반환하는 지연 호환 계층을 사용합니다.
- 새로운 DP3 패키지는 `paths.snapshot_library()`로 전체 라이브러리를 복사합니다. 기존 frozen snapshot을 재사용하는 확대 경로는 그대로입니다.
- 특정 원격 snapshot에 고정된 이전 x0/low-noise/HOPE 검사·비교 실행기의 내부 경로는 유지했습니다. 일반적인 신규 실험 실행기가 아닙니다.

## 검증

- 전체 테스트 **724 passed** (`tests-final.log`). upstream Dexonomy의 기존 NumPy 경고 138건이 남습니다.
- CLI 29개가 저장소 밖 작업 디렉터리에서 `--help` 실행에 성공했습니다. 나머지 이동된 Python 파일 2개는 CLI가 아닌 모듈입니다.
- 이동한 셸 스크립트 5개 `bash -n` 통과.
- 실제 보관 시범 `AlphabetSoup-409a5d4a75d5`를 새 라이브러리로 1,024 tick 재생했습니다. 관절 상태와 접촉·품질 신호 10종의 최대 오차가 모두 0이며, 기존 성공 판정도 같습니다. 새 후보를 실행한 것이 아니라 구조 변경 회귀 검사입니다.
- import 순서 양방향의 모듈 객체 동일성, import 지연, 옛 provenance 키 보존, 저장소와 분리한 코드 snapshot import를 회귀 테스트로 확인했습니다.
- 원장 전체 흐름 테스트는 가짜 물리를 사용하는 기존 테스트입니다. 역사적 데이터에 붙은 코드 해시를 가짜 실행의 기준 해시로 주입했으며, 실제 실행기의 엄격한 해시 검사는 완화하지 않았습니다.
- Ruff: 이전 snapshot 1,096건 → 현재 1,036건. 경로 대응 후 코드·메시지별 대조에서 새 항목 0건입니다. 전체 lint가 깨끗하다는 의미는 아니며 기존 문제는 별도입니다.
- `git diff --check` 통과, 활성 문서 링크 96개 모두 유효합니다.
- import·경로 이동을 정규화한 AST 대조에서 공용 모듈 26/27개의 나머지 코드는 같습니다. 나머지 `student_inference`는 과거 소스 해시 키 조회 호환을 추가한 차이입니다. 제어·보상·품질 기준을 바꾸지 않았습니다.

## 보존과 복원

- `before.tar.gz`: 작업 시작 당시 코드·테스트·문서 snapshot. 기존 사용자/Claude의 미커밋 변경도 포함합니다.
- `before-hashes.json`, `moves.json`, `modules.json`, `document-moves.json`: 원본 해시와 이동 대응.
- `frozen-launcher-exceptions.json`: 고정 원격 경로를 유지한 파일 목록.
- 실험 결과·out 파일은 snapshot 이후 수정 목록이 모두 비어 있습니다. 시범·checkpoint·HANDOFF·모델·원격 실행 사본을 수정하지 않았습니다.
- `results`, `scratchpad` 호환 링크와 기존 out 데이터 경로를 유지했습니다.
- 엄격한 예전 실험 재현은 당시 원격 코드 사본을 사용합니다. 옛 import가 작동한다고 변경된 코드 해시까지 같아지는 것은 아닙니다.
- 복원은 snapshot과 이동 목록을 기준으로 필요한 파일만 수행합니다. Git 전체 reset/clean은 하지 않습니다.

전체 구조는 [docs/layout.md](../../../docs/layout.md), 최신 재개 위치는 [RESUME](../../../RESUME.md)를 참고하세요.
