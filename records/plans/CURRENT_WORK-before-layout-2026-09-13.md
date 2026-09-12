# 현재 작업

실행 상태의 기준은 [RESUME](../../RESUME.md), 범위·시간 예산은
[24시간 계획](../../plans/SPRINT_24H_2026-09-12.md), 공용 파일 고정은
[구조 정리 공지](../../plans/REFACTOR_COORDINATION_2026-09-12.md)입니다.

- Codex: 시각 데이터 인수 → 행동 오차·복귀 데이터 진단 → 조건을 갖춘 lecun BC 비교.
- Claude: 같은 장면·같은 초기 로봇 상태에서 목표를 바꾼 teacher 시범.
- 구조 정리: 비활성 코드/문서부터 진행. 두 실행이 사용하는 파일은 내용·경로 고정.

기존 상세 진행 내용은 [이전 작업서](CURRENT_WORK-before-layout.md)에 보관했습니다.
완료된 실험을 과거 계획만 읽고 재시작하지 않습니다.

## 2026-09-12 23:03 — 별도 1단계 DP3 실험

- lecun GPU3 / tmux `codex-stage1-dp3`; Claude BC/PPO는 GPU2.
- [고정 계획과 상태](../experiments/2026-09-12/stage1-dp3-feasibility/README.md).
- 현재 형상과 Dexonomy 목표를 받는 실행 모델. 3물체 각1시범, 6000갱신 후9회 평가.
- 원본 재생3/3 모든 신호 오차0 후 학습 시작. 시각 VLA나 미지 형상 성공으로 해석하지 않음.
- 실행 사본 `/data1/yspark/projects/dexprompt/stage1-dp3-v1/package/code` 변경 금지.

### DP3 완료 업데이트 (23:14:12 KST)

학습 6000갱신 및 평가9회 종료. AlphabetSoup 2/3, Peaches 0/3, Raisins 3/3으로 합계5/9 완주.
학습한 목표 조건의 전체 팔·손 실행 검증이며, 새 형상/목표 선정/VLA 결과는 아니다. 추가 학습은 시작하지 않았다.

## 2026-09-13 — DP3 다배치 확대 실행

[고정 분할·실행 계획](../experiments/2026-09-13/stage1-dp3-placement/README.md).
lecun `codex-dp3-placement`, GPU3. 72시범 중 학습55/미학습배치17, 진단6조건×3회.
기존 75개 재생 상태 재사용, 20,000갱신, 최종EMA만 평가. 학습조건 완주0이면 확대 평가 중단.
고정 실행 사본 변경 금지. Claude MLP/공용 코드·기존 자료는 유지한다.

### DP3 다배치 완료 (01:03:20 KST)

대조군23/23, 학습조건3/18, 미학습배치0/51. 전 단계 정상 종료.
세션 `codex-dp3-placement` 종료; 후속 자동 학습 없음. 최종 결과는 위 실험 보고서 참조.
