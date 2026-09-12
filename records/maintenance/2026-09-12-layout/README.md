# 구조 정리 1차 완료

2026-09-12. [현재 구조](../../../docs/layout.md).

## 변경

- scripts 최상위 Python 147개 중 비활성 117개를 기능별 8폴더로 이동.
  현재 최상위는 실행 중 작업이 쓰는 30개와 공통 실행기·bootstrap 2개다.
- tests 86개를 6분야로 이동. Claude/Codex가 고정한 테스트 3개와 conftest는 최상위 유지.
- 종료된 계획·인계 문서와 설치/계약 문서 24개를 records/plans 및 docs로 분류.
- 루트 Markdown은 README, RESEARCH_PLAN, PAPER_TODO, RESUME 4개로 축소.
  이전 긴 RESUME/CURRENT_WORK는 records/plans에 보관하고 현재 문서는 짧게 정리.
- results → records/experiments, scratchpad → records/scratchpad로 이전.
  예전 두 경로는 상대 심볼릭 링크로 유지한다. 원시 결과·실패 기록은 삭제하지 않았다.
- out 최상위의 비활성 실행 스크립트·로그 30개를 records/scratchpad/out-root로 이동.
  데이터셋·manifest·체크포인트·자산과 실행 중 결과 경로는 유지.
- 옛 CLI는 `python scripts/run.py old_name.py ...`로 실행한다.
  직접 옛 파일 경로를 호출하는 명령은 이 진입점 또는 새 경로로 변경해야 한다.

## 호환성·검증

- 변경 전 원본 snapshot: before.tar.gz (코드와 문서; 사용자 기존 수정 포함).
- 코드/문서 이동 목록: moves.json, 옛 CLI 대응: ../../../scripts/legacy_paths.json.
- 고정한 스크립트·src·테스트·설정 70파일의 해시 변화 0. protected.json에 목록이 있다.
- 이동 전후 기존 테스트 수집은 696개로 동일했다.
- 처음 전체 검사에서 동적으로 조합하던 옛 파일 경로 때문에 오류가 났다.
  경로 resolver를 적용했고 해당 검사 및 CLI 회귀 44개 통과.
- 최종 전체 테스트 **701개 통과** (기존696 + 새 CLI/경로 회귀5), exit 0.
- 주요 CLI 4개 --help 통과, 다른 cwd에서 직접/옛 이름 실행 회귀 통과.
- 활성 문서의 누락된 로컬 Markdown 링크 0.
- Ruff는 로컬 설치가 없어 실행하지 않았다. 전체 변경의 lint 통과를 주장하지 않는다.
- 위 검사는 코드 이동의 회귀 검사다. 새 학습이나 파지 성능 개선을 의미하지 않는다.

## 남은 구조 이전

Claude target-switch와 Codex visual-intake가 끝나기 전에는 src/icra와 고정된 30개 스크립트를
이동하지 않는다. src의 분류안은 docs/layout.md에 있다. 자동 이전 작업은 아직 예약하지 않았다.
코드 변경과 결과 해시 재현을 구분하고 과거 코드 snapshot은 그대로 사용한다.

기록: collection-before/after.log, tests-after.log(초기 경로 오류), path-regressions.log,
tests-final.log(최종 통과), cli-checks.json, active-links.json, out-loose-files.json.
