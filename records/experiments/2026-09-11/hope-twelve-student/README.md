<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 승인된 HOPE 12시범 통합과 A/B 자동 보고

2026-09-11 23:25 KST, lecun의 별도 tmux 두 개를 시작했다. **검사 진행 중**이다.
현재 low-noise 학습의 모델·코드·입력과 기존 source 데이터는 수정하지 않는다.

## 입력과 완료 조건

- 기존 fixed7(7,466프레임)과 인수 완료한 새5(5,011프레임), 합계12/12,477.
- 3물체, teacher 후보5개. 전부 A 물리 조건이고 Milk B는 제외한다.
- 각 release에 고정된 manifest 해시·학생 인수·대조군 성공을 검사한다.
- 기존 준비 도구로 새 묶음을 생성하고 모든 NPZ 멤버의 압축 해제 바이트를 원본과
  대조한다. 물리 신호도 포함하며, 재생용 모델·초기 상태·코드 출처는 해시 검증 후 복사한다.
- 새 cache, 입력 감사, train-only preflight를 수행한다. 기존 recorded/stationary 대조군은
  동일 자료의 검사 근거로 보존하며 다시 시뮬레이션하지 않는다.
- 새 후보2개 각각 tick0/500으로 실제 DINO full-model 검사2배치.
  유한 손실·행동, 모델 상태 불변, backward/optimizer0을 모두 요구한다.
- 전체 통과 후에만 release.json을 쓴다. 실패 시 stage exit/timeout 기록 후 중지한다.
- 새 학습이나 독립 validation/test는 없다. 일반화 성능을 주장할 자료가 아니다.

## lecun 작업 위치

| 작업 | tmux | 결과 |
| --- | --- | --- |
| 12시범 통합·검증 | `dexprompt-hope-twelve-v1` | `/data1/yspark/projects/dexprompt/student-checks/hope-twelve-job-v1/checks` |
| low-noise A/B 읽기 전용 감시 | `dexprompt-low-noise-report-v1` | `/data1/yspark/projects/dexprompt/student-checks/low-noise-comparison-v1` |

통합 코드 snapshot: `/data1/yspark/projects/dexprompt/student-checks/hope-twelve-job-v1/code`.
새 데이터: `/data1/yspark/projects/dexprompt/student-data/hope-twelve-v1`.
새 cache: `/data1/yspark/projects/dexprompt/student-cache/hope-twelve-v1`.
CPU 라이브러리1thread, nice15. DINO forward는 빈 GPU3에서 메모리 점검 후 수행한다.
기존 학습 GPU1은 그대로 둔다. 원격 `status.json`을 먼저 읽고 중복 실행하지 않는다.

A/B 보고는 epoch1/3/6/9/13의 동일 seed1000–1002와 모델·초기 상태·평가코드·데이터
일치를 검사한다. 최종은 사전고정epoch13이며 중간checkpoint 최고값으로 바꾸지 않는다.
누락 평가는 미완료로 남긴다. 인프라 오류를 파지 실패0건으로 대체하지 않는다.
최종24개 teacher 관측의 첫8팔행동 RMSE를 별도로 비교한다. 이 오차 감소가
폐루프 파지 성공을 의미하지는 않는다. weighted train_loss의 크기로 A/B를 비교하지 않는다.

## 코드 검사

새 도구2개와 기존 실행계약 관련 테스트 합계8개 통과, 변경파일 Ruff 통과.
원본 release에서 7+5 승인 확인, 미승인/다른manifest 거부, NPZ 신호 변경 검출,
미완료 평가·중복seed·평가계약 변경·실행오류 처리와 seed별 성공 비교를 검사했다.
전체 suite는 이번 작업에서 실행하지 않았다.

[역할 분담과 Claude 전달문](../../../plans/FAST_PARALLEL_NEXT_2026-09-11.md).

학습 상태 추가 확인: low-noise 학습은 **6,331/6,331 갱신을 완료**했고,
`evaluate-01`로 자동 전환했다. 마지막 가중 train_loss 0.00123087은 실행 기록일 뿐
기존 무가중 손실과 직접 비교하지 않는다. 폐루프 최종 성공 수는 아직 미완료다.
