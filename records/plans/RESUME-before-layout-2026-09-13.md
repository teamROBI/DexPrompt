# 현재 실행과 재개

2026-09-12. [24시간 계획](../../plans/SPRINT_24H_2026-09-12.md)을 기준으로 진행합니다.
과거 진행 로그는 [records/plans](.)에 보관했습니다.

## Codex — student 인수

- tmux: `codex-followup-intake` (Turing).
- 출력: `/data1/yspark/sim/icra/followup-visual-intake-20260912-v1`.
- 먼저 `intake/status.json`, `intake/results.json`, `launcher-exit.json`을 읽습니다.
- teacher 49건의 전수 파일 검증 통과. 접촉 검사 47건 통과·2건 보류.
- 신규 44건 중 접촉 검사 통과는 42건입니다. 시각 인수와 기존 자료 중복은 별도로 판정합니다.
- 순서: 재렌더 → cache → 시각 입력 감사 → 기록 행동 대조군 → preflight.
- 기존 A12에 병합하지 않았고 새 학습도 시작하지 않았습니다. deadline 미처리는 pending으로 남습니다.
- [실행 기록](../experiments/2026-09-12/followup-visual-intake/README.md).

## Claude — 다물체 목표 변경

- tmux: `icra-target-switch`, 출력 `/data1/yspark/sim/icra/target-switch-v1`.
- 3배치·2목표·목표별 최대2후보, 물리 상한20회. 같은 초기 상태 계약을 먼저 검사합니다.
- 보고 마감은 2026-09-13 00:05 KST. 현재 단계는 실제 status/ledger를 읽어 확인합니다.
- [공용 코드 고정 공지](../../plans/REFACTOR_COORDINATION_2026-09-12.md).

## 코드 구조 변경

- 비활성 scripts/tests를 기능별로 이동했습니다. 이전 명령은 `scripts/run.py old_name.py`로 실행합니다.
- 두 실행이 사용하는 스크립트30개, src/icra 및 관련 설정은 경로와 파일 내용을 보존합니다.
- [구조·이전 규칙](../../docs/layout.md), `records/maintenance/2026-09-12-layout/protected.json`을 확인합니다.
- results와 scratchpad는 records 아래의 기록을 가리키는 호환 링크입니다.
- 과거 out 폴더3개는 /data1로 옮겼고 기존 경로는 유지합니다. [공간 정리](../experiments/2026-09-12/workspace-cleanup/README.md).

## 완료된 작업 — 재시작하지 않음

- [공식 행동 backend 비교](../experiments/2026-09-12/upstream-action-feasibility/FINAL.md): 6,331갱신 후 최종0/3, 접촉 획득0.
- [teacher 배치 인계](../experiments/2026-09-12/teacher-followup-handoff/README.md): 49건 재현, 정확한 중복5·신규44.
- [선정기 logit 감사](../experiments/2026-09-12/selector-logit-audit/README.md): 수치 포화 정정. 같은 평가 풀의 사후 비교와 독립 검증을 구분합니다.
