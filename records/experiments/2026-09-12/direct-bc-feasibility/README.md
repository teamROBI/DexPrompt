# 당일 시각 BC 실행 가능성 비교

2026-09-12, 시작02:58 KST / 전체 작업 종료 상한09:58 KST.
사용자는 추가 승인 없이 7시간 안에 가능성을 판단하도록 요청했다. 학습은 lecun에서만 수행한다.

## 사전 고정

- 기준선: `student-runs/single-bottle-x0-v1`, 최종epoch13, 완주0/3.
- 비교: 같은 conditional Transformer·실제 frozen DINO·데이터·정규화·horizon64·execute8·seed42·13epoch/6331갱신.
- 변경 요인: 확산 x0 예측 → deterministic direct action chunk regression.
  입력은 고정 zero query/time0이며 정답 행동은 loss에만 사용한다. masked normalized action MSE.
  추론은 Transformer1회, 출력 범위 clipping/역정규화는 기존과 같다.
- 직전 low-noise B와의 복합 변경을 단일 요인이라 하지 않는다. 비교 기준은 가중치 없는 x0 A다.
- 평가: 사전 고정 최종`epoch_0012.pt`, seed1000–1002, 같은 보관 시작 상태·접촉 획득·hold·품질 기준.
  direct 정책은 결정론적이므로 세 번은 독립 성공률 표본이 아니다.
- 알려진 학습 과제 진단이다. 미지 물체·지시 일반화와 논문 전체 성공의 증거가 아니다.

## 실행

- tmux: `dexprompt-direct-bc-v1`, GPU2. 기존 다른 GPU 작업은 건드리지 않았다.
- 코드: `/data1/yspark/projects/dexprompt/student-code/direct-bc-v1`
- 작업: `/data1/yspark/projects/dexprompt/student-checks/direct-bc-v1`
- 모델: `/data1/yspark/projects/dexprompt/student-runs/direct-bc-v1`
- 기준선 코드 사본에서 student만 교체했다. 원본 모델/데이터/코드 보존.
- 단계: preflight→기록명령/정지 대조→학습→최종 폐루프 평가. 단계 실패/timeout이면 후속 중지.
- 학습 최대3시간, 평가30분, 전체 deadline 상한. stage PID/exit/timeout은`stages.json`에 기록.
- 최초 관련 회귀검사45개 통과(직접예측·x0·low-noise·checkpoint/inference).
  로컬 Ruff 실행 파일은 없어 lint는 실행하지 못했다. syntax compile 통과.

## 판정

원격 preflight·대조군 통과. 첫 direct BC는 최종 평가까지 완료했고 완주0/3이었다.
조건부 residual BC도 최종 평가까지 완료했고 완주0/3이었다. 확정 결과는 아래와 최종 보고서에 기록한다.
성공 시 다음은 여러 시범/배치에서의 전이 확인. 실패 시 최초 명령 오차와 이후 이탈을 짧게 비교하고,
같은 장시간 학습을 반복하지 않는다. Claude 편집 결과와 시각 결과는 별개로 최종 판단한다.

## 추가로 고정한 조건부 후속

관절 상태 기반 최근접 시범 재사용 대조는 완주했다(14.64cm, 2/2접촉군, 유지2.76초).
이것은 한 학습 궤적을 비시각적으로 재사용한 결과이며 시각 학습·새 상태 복귀·일반화 근거가 아니다.
시계/물체 pose/힘/teacher phase를 검색 입력으로 쓰지 않았다.

첫 direct BC의 최종 평가가 정상 완료되면서 완주0일 때만 동일13epoch/6331갱신의
state-residual direct BC를 진행한다. 직접 BC가 성공하거나 인프라 실패하면 후속 학습은 건너뛴다.
변경은 출력 매개화 하나다: 현재 관측 관절값에서 actuator 목표까지의 차이를 예측하고
현재 관절값을 더해 기존과 동일한 절대 명령으로 전달한다. 중력 보정은 추가하지 않는다.
현재 상태는 raw 관측에서 가져오며 teacher 목표/미래 상태를 사용하지 않는다.
출력 clipping은 residual이 아니라 복원된 absolute action 범위에 적용한다.

- 후속 대기 tmux: `dexprompt-residual-conditional`
- 별도 코드/작업/모델: 각 `residual-direct-bc-v1` 디렉터리. 현재 direct 실행 snapshot은 변경하지 않았다.
- 관련 검사47개 통과. 후속도 동일09:58 KST 종료 상한을 적용한다.

## 자동 최종 판정

Turing tmux `codex-feasibility-watch`가 원격 두 작업과 Claude 편집 완료 요약을 읽는다.
둘이 끝나면 즉시, 늦어도09:58 KST에 `../day-feasibility/FINAL.md`를 작성한다.
인프라 실패는 완주0으로 바꾸지 않는다. 단일 과제 성공과 원래 논문의 전체 검증을 구분한다.
감시자 판정 회귀검사3개 통과. 인터랙티브 세션을 계속 호출하지 않아도 작업과 집계는 진행된다.

## 첫 비교 확정

Direct Transformer BC 최종13epoch/6331갱신: 완주0/3, 접촉 획득0, 목표 접촉군 최대0.
첫 팔 명령 최대 오차0.025523rad, RMSE0.014018rad. 초기 오차 감소가 접촉 성공으로 이어지지 않았다.
사전 조건에 따라 residual BC가 시작됐다.
다물체 시작 자세의 범위는 `../day-feasibility/RESET_SCOPE.md`에 별도 기록했다.

## 조건부 후속 확정 및 종료

Residual direct BC도 13epoch/6,331갱신을 완료했다. 마지막 기록 loss0.000580344, 최종 완주0/3,
접촉 획득0/3, 지속 리프트0/3. 세 실행 모두974tick, 상승4.9246e-7m였다.
first-action 팔 최대 오차0.091351rad, RMSE0.041837rad.
두 실행 모두 단계 오류 없이 완료됐고 관련 원격 tmux는 종료됐다.
추가 단일 과제 튜닝은 하지 않는다. [최종 판단과 역할별 다음 작업](../day-feasibility/DECISION.md).
