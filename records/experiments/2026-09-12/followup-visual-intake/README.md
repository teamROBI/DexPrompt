# 배치 확장 시범의 student 인수

2026-09-12. 학습 없이 별도 인수를 시작했다. 기존 A12, teacher 원본과 HANDOFF는 수정하지 않는다.

## 확인한 입력

- teacher-followup HANDOFF: 49건, 3물체, 6후보, 34개 물체–배치 조합.
- 원장 136회는 실행 종료 수이며 파지 성공 수가 아니다. 배치 실행 72회 중 49회 accepted.
- 49건의 원본·재현 episode와 초기 상태 파일 해시를 모두 직접 확인했고 불일치가 없었다.
- 기존 AcquisitionGuard로 보관 신호를 검사한 결과 47/49 통과.
  AlphabetSoup 2건은 `approach_displacement`로 보류했다. 이는 teacher 판정 철회가 아니라
  student 접촉 획득 계약의 추가 검사다. 통과 47건도 아직 student-ready가 아니다.
- 기존 자료와의 중복·순증분은 Claude의 별도 감사 대상이다. 기존 시범 수에 49를 더하지 않는다.

## 실행

- tmux: `codex-followup-intake` (Turing).
- 출력: `/data1/yspark/sim/icra/followup-visual-intake-20260912-v1`.
- 생성 스크립트: `scripts/prepare_followup_intake.py`.
- 인수: 기존 `scripts/intake_hope_diversity.py`의 새 `--frozen-plan` 경로.
- 준비에서 모든 원본·재현의 episode/model/initial-state 해시와 후보·배치·manifest를 검증한다.
  오류는 항목별로 기록하고, 증거 오류가 있으면 자동 인수를 시작하지 않는다.
- 접촉 검사를 통과한 항목은 후보별 한 건씩 먼저 처리한다. 각 후보 안에서는 이미 기록된
  품질 여유 내림차순이다. 결과를 보고 순서를 바꾸지 않으며 이 순서는 평가 분할이 아니다.
- 순차 실행: 전체 물리 기록을 대조하며 visual-v2 재렌더 → cache → 시각/목표 기억 감사
  → 기록 행동 양성·음성 대조군 → preflight.
- GPU 0은 EGL 재렌더에만 사용하고 학습은 실행하지 않는다. 단일 작업, BLAS 1 thread,
  nice 15. 작은 EGL 렌더러 검사는 통과했다.
- 준비 상한 15분, 인수 상한 3시간. 남은 항목은 `pending_count`로 남긴다.
  파일 공간이 20 GiB 미만이면 해당 인수를 실패로 기록한다. 임시 파일도 새 /data1 출력 아래 둔다.
- 데이터 검사를 모두 통과해도 여기서는 실제 DINO forward나 student 파지 성공을 검증한 것이 아니다.

## 상태 확인과 재개

`preparation/guard-audit.json`은 전수 파일·접촉 검사,
`preparation/intake-plan.json`은 고정된 입력 순서,
`intake/status.json`은 현재 단계, `intake/results.json`은 완료·실패·미처리 수다.
각 단계의 로그와 `*.finished.json`에 종료 코드·timeout·시간을 남긴다.
`launcher-exit.json`은 실행기 종료 코드다. 실행기 exit 0과 모든 항목 인수 통과는 다르다.

중복 실행하지 않는다. 중단이 필요하면 `intake/STOP`을 만들면 현재 항목 처리 후 멈춘다.
완료 후 미처리 항목은 기존 결과를 보존한 별도 계획/출력으로 재개한다.
이 스크립트는 자동 학습이나 기존 데이터 병합을 하지 않는다.

## 검사와 남은 일

관련 테스트 12개 통과. 셸 구문과 수정한 기존 파일의 공백 검사 통과.
Ruff는 로컬 환경에 설치돼 있지 않아 실행하지 못했다. 전체 suite는 실행하지 않았다.

인수 완료 후 Claude 중복 원장과 연결하고, 후보/배치 누설을 막는 학습·평가 분할을
별도 정의한다. 배치가 다른 시범은 로봇이 시범 경로에서 벗어났을 때의 복귀 시범과 다르다.
직전 student 공식 backend 비교는 0/3이므로, 데이터 증가만으로 폐루프 성공을 가정하지 않는다.
