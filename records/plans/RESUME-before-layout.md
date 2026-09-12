> **현재 실행: 배치 확장 student 인수(2026-09-12 저녁).**
> Turing tmux `codex-followup-intake`, `/data1/yspark/sim/icra/followup-visual-intake-20260912-v1`.
> [실행/재개 기록](results/2026-09-12/followup-visual-intake/README.md). status/results/launcher-exit부터 읽고 중복 실행하지 않는다.
> teacher49건 중 접촉 검사47통과·2보류. 파일 검증→시각 재현→입력 감사→대조군→preflight를 순차 수행한다.
> 새 학습·기존 A12 병합 없음. Claude는 별도 중복 감사를 진행한다.
> 저장 공간 정리 완료: 과거 `out/teacher-expansion`, `out/offset-compare-93`, `out/candidate_screen_all`은 /data1로 이동했으며 기존 경로는 symlink로 유지한다. [908파일 해시 검증·정리 기록](results/2026-09-12/workspace-cleanup/README.md).

> **후속 3시간 비교 완료(14:13 KST):** 공식 행동 backend6331갱신·최종0/3, 접촉 획득0.
> [결과](results/2026-09-12/upstream-action-feasibility/FINAL.md). 전용 lecun tmux 종료. 아래 실행 중 표기는 과거 기록이다.
> Claude 선정 평가 중 발견한 수치 문제는 [logit 감사](results/2026-09-12/selector-logit-audit/README.md). 고정 순위는 변경하지 않았다.

> **새 3시간 작업 시작(2026-09-12 12:59–15:59 KST):** lecun tmux `dexprompt-upstream-action-v1`.
> 공식 행동 Transformer/DDIM/optimizer 경로 한 건,13epoch/6331갱신→최종3회 평가 자동 연결.
> `student-checks/upstream-action-v1/status.json`을 먼저 확인하고 중복 실행하지 않는다.
> Turing `codex-upstream-watch`가 결과 회수/한국어 `results/2026-09-12/upstream-action-feasibility/FINAL.md` 자동 작성.
> [실험 계획](results/2026-09-12/upstream-action-feasibility/README.md). 아래는 이전05시 작업의 결론이다.

# 재개 지점 — 2026-09-12 최종 평가 완료

먼저 [최종 판단](results/2026-09-12/day-feasibility/DECISION.md),
[진행판](PAPER_TODO.md), [현재 작업서](plans/CURRENT_WORK.md)를 읽는다.

사용자 승인 7시간 작업(02:58–09:58 KST)은 제한 안에서 비교를 완료했다.
새 direct BC와 residual BC는 각각6,331갱신/최종 완주0/3. 기존x0/low-noise도0/3.
지정 lecun 학습·평가 tmux는 종료 확인. 완료된 작업을 재시작하지 않는다.
최근접 관절 상태 시범 재사용은 같은 학습 궤적에서 성공했고, 시각/일반화 증거는 아니다.
Claude 손목편집24회 개선0, zero-edit6회 재현.
HOPE A12 통합 검사 완료(12시범/12,477프레임/3물체/5후보), 새 다물체 학습은 하지 않았다.

## 연구 방향

1. 정확한 형상·배치·로봇 상태를 이용한 Dexonomy 파지 선정/시범 적응 학습과 자동 수집.
2. 인수된 시범으로 DexGraspVLA 방식의 목표 조건부 시각 행동 BC.
부위 지시는 필수가 아니다. mask·DINO는 DexGraspVLA 원 설계와 양립한다.
고정 후보 팔 접근 PPO의 결과를 1번 전체의 실패로 해석하지 않는다.

## 결정과 다음 작업

현재 student 구현의 대규모 확대와 같은 고정 과제의 반복 튜닝을 보류한다.
아이디어 자체의 불가능을 판정한 것은 아니다. 다음은 검증된 행동 학습 구현과 최소 재현 비교,
teacher 측은 손 모양을 포함한 편집 이식 계약이다. 자세한 종료 조건은 최종 판단 문서를 따른다.
유형별 초기 손 자세 범위는 `results/2026-09-12/day-feasibility/RESET_SCOPE.md`.

## 보존과 실행 경로

lecun 프로젝트 `/home/yspark/projects/DexPrompt`, 대용량 루트 `/data1/yspark/projects/dexprompt`.
결과는 `student-checks/{direct-bc-v1,residual-direct-bc-v1,proprio-retrieval-v1}`.
모델은 `student-runs/{direct-bc-v1,residual-direct-bc-v1}`, 각 최종epoch13.
원본/별도 코드 snapshot/체크포인트 보존. 물리·접촉 기준은 변경하지 않았다.
Milk B와 신규 teacher/clutter 자료는 기존A12에 자동 합치지 않았다.
학습은 lecun, 긴 작업은 tmux. 다른 사용자 작업은 건드리지 않는다.

SSH agent 경로는 세션마다 바뀔 수 있다. 현재 연결에 사용한 소켓은
`/tmp/ssh-XXXXXXh9aveI/agent.3844836`, identity는 `~/.ssh/id_rsa_yspark`.
연결 실패를 실험 실패로 세지 않는다. 자세한 원격 상태는 최종 snapshot에 있다.
