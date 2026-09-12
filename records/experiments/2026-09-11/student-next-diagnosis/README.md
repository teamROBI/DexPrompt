<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# Student G0→G3: 폐루프 이탈 진단, HOPE 통합, 다음 실험

2026-09-11. 기존 단일 병 x0 최종 체크포인트와 저장된 세 평가 실행만 분석했다.
새 정책 forward, backward, optimizer 갱신, teacher 실행은 없었다. 원시 계산 결과는
[`archived-analysis-v2/diagnosis.json`](archived-analysis-v2/diagnosis.json), 그림은
[`archived-analysis-v2/trajectory.png`](archived-analysis-v2/trajectory.png)에 있다.

## G0. 기준 고정

실행 전에 [`plan.json`](plan.json)에 x0 최종 checkpoint, epsilon 기준 checkpoint,
단일 병 manifest, 최종 평가, 명령 감사와 HOPE 고정 pool의 SHA256을 기록했다. 로컬과
lecun의 디스크·GPU 상태, 새 forward 상한, 새 학습 금지도 함께 고정했다. 이 파일은 후속
결과로 다시 쓰지 않았다.

## G1. x0 student의 최초 폐루프 이탈

### 결론

학생은 **낯선 폐루프 관측을 받은 뒤가 아니라, 시범과 완전히 같은 최초 입력에서 낸 첫 팔
명령부터 벗어난다.** 이후 8 tick마다 다시 예측할 때 생기는 큰 명령 불연속이 오차를 더
키운다. 따라서 현재 근거는 입력 복원·물리 실행 오류보다 **정책의 최초 action 생성 오차와
재계획 연결 방식**을 먼저 검사해야 한다는 쪽을 지지한다.

이 결과만으로 DINO 표현, 데이터 양, diffusion 목적함수 중 어느 것이 최초 action 오차를
만들었는지는 구분할 수 없다. 최근접 teacher 상태도 보조 진단일 뿐 경로 유효성 판정으로
사용하지 않았다.

### 최초 입력과 최초 명령

평가가 저장한 `initial_observation.npz`를 원본 시범 tick 0과 비교했다. 머리 RGB+mask,
손목 RGB, 관절 상태, object/part mask, 가시성·pixel 수의 **8개 배열이 모두 bit-identical**이고
최대 절대 차이는 0이다. 감사 기록에서도 같은 입력과 전처리 tensor가 일치한다.

| 항목 | seed 1000 | seed 1001 | seed 1002 |
| --- | ---: | ---: | ---: |
| tick 0 student−teacher 최대 명령 오차 | 0.1628 rad | 0.1698 rad | 0.1575 rad |
| 해당 관절 | arm joint 1 | arm joint 1 | arm joint 1 |
| tick 0 student 명령−측정 상태 최대값 | 0.1715 rad | 0.1785 rad | 0.1662 rad |
| teacher 명령−측정 상태 최대값 | 0.0282 rad | 0.0282 rad | 0.0282 rad |
| 상태 오차가 0.1 rad를 처음 넘은 tick | 11 | 11 | 12 |

첫 64 tick의 arm 명령 RMSE는 0.189–0.199 rad이고 hand는
0.000234–0.000251 rad이다. 이 구간의 최대 오차 0.716–0.743 rad도 tick 56의
arm joint 1이다. 즉 최초 실패는 손가락 접촉 제어가 아니라 접근 팔 명령에서 시작한다.

### 8-tick 재계획 경계

teacher의 첫 64 tick은 경계와 구간 내부의 정규화 RMS 명령 변화 중앙값이 각각
0.005291과 0.005287로 같다. 반면 student는 다음과 같다.

| seed | 경계 중앙값 | 구간 내부 중앙값 | 비율 | 첫 64 tick 최대 변화 |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 0.05864 | 0.00220 | 26.6× | 0.227 rad, tick 48 |
| 1001 | 0.06652 | 0.00200 | 33.3× | 0.251 rad, tick 48 |
| 1002 | 0.06443 | 0.00209 | 30.9× | 0.257 rad, tick 48 |

세 최대값은 모두 재계획 경계의 arm joint 1에서 나왔다. 다만 첫 명령 자체가 이미 틀리므로
경계 불연속은 **최초 원인**이 아니라 반복적으로 오차를 키우는 두 번째 현상이다. 이번
진단에서는 rate limiter나 temporal ensemble을 붙이지 않았다.

### 단순 지연인지 여부

훈련의 `state_scale`로 정규화한 뒤 arm 7관절과 hand 20관절을 각각 같은 0.5 가중치로
두고, 매 student 상태에 가장 가까운 원본 teacher 상태 974개를 찾았다. student tick 0은
teacher tick 0과 정확히 같지만 tick 64는 teacher tick 113–115, tick 200은
teacher tick 369–411에 가깝다. 초반에는 teacher 경로를 지나치게 빨리 진행한다.

이후 최근접 teacher index는 실행별 최대 546/682/657까지만 갔다가 종료 시
533/523/657에 머문다. 전체에서 동일 tick ±8 안에 있는 비율은 2.1–3.0%이고,
최근접 index가 단조 증가하는 비율도 83.0–86.6%다. 그러므로 하나의 고정된 시간 지연으로
설명되지 않으며, 중후반에는 정지·후퇴·다른 상태가 섞인다.

물리 결과도 이 해석과 일치한다. 세 실행의 최대 동시 접촉군은 0/1/1(필요 2), 모든 군이
동시에 성립한 tick은 0, 최대 상승은 0.0004/0.998/0.996 mm다.

### G1-B를 실행하지 않은 이유

계획서의 고정 관측 추가 forward는 기존 자료만으로 시작점이 불명확할 때 수행하는 선택
검사였다. 여기서는 최초 정책 입력 전체가 원본과 같고 상태 오차 0인 tick 0에서 이미 세
seed 모두 0.158 rad 이상 틀렸다. 기존 4-frame forward 감사와 같은 질문을 최대 48회
반복해도 이 분기를 바꾸지 않으므로 실행하지 않았다. 새 forward 수는 0이다.

### 남는 가설

- **지지됨:** 최초 절대 action chunk 생성 오차. 동일 시작 입력에서도 팔 첫 명령이 teacher와
  크게 다르다.
- **지지됨:** chunk 간 연결 불연속. teacher에는 없는 큰 변화가 8-tick 경계마다 반복된다.
- **반대 근거가 큼:** 폐루프 렌더·전처리·명령 재생 오류가 최초 이탈을 만들었다는 설명.
  최초 입력은 동일하고 이전 명령 물리 재생도 신호 차이 0이었다.
- **미분리:** DINO 특징, action 회귀/확산 목적, 절대 action 표현, 학습 표본 수 중 무엇이
  최초 출력 오차를 지배하는가. 이번 결과만으로 비전 grounding이나 일반화를 판정하지 않는다.

분석 스크립트와 경계·가중치·최초 입력 회귀 테스트 4개는 통과했고 Ruff도 통과했다.

## G2. 승인된 HOPE 7시범 통합과 lecun 연결

고정 입력은 `out/hope-student-pool-final.json` 하나만 사용했다. 새로 readiness를 계산한
결과도 원본과 SHA256 `f13daac296a4a851…`로 같았고, 빠진 시범을 다른 후보로 대체하지
않았다. 통합 결과는 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| 시범 / 프레임 | 7 / 7,466 |
| 물체 / 물체-배치 조건 | 3 / 7 |
| 고유 teacher 후보 / 파지 유형 | 3 / 2 |
| 고유 최초 정책 입력 | 7 |
| 독립 validation / test | 0 / 0 |

로컬 묶음은 `/data1/yspark/sim/icra/hope-student-bundle-v1`이고 manifest SHA256은
`d9e98792ad218562…`다. 원본에서 통합본으로 옮긴 7개 episode의 action, state, 품질,
조건, 모델과 초기 상태 해시를 전부 대조했다. 로컬 cache·인과적 시각 입력 감사·preflight를
통과했고, 기록 행동 7회는 모두 teacher-compliant lift, 정지 대조군 7회는 모두 비성공이었다.

전송은 156파일, 1,653,463,986바이트를 lecun의
`/data1/yspark/projects/dexprompt/student-data/.hope-bundle-v1.partial`에 먼저 복사했다.
전후 전체 파일 해시를 확인한 뒤
`/data1/yspark/projects/dexprompt/student-data/hope-bundle-v1`로 원자적으로 바꿨다.
원격 cache·감사·preflight와 같은 7+7 물리 대조군도 통과했다.

실제 DINO 전체 모델 검사는 각 물체의 첫 시범에서 home과 첫 close 표본을 한 장씩 골라
정확히 3배치만 수행했다.

| 물체 | 표본 index | 출력 | 유한 loss/action | 모델 불변 | 학습 호출 |
| --- | --- | --- | --- | --- | --- |
| Peaches | 0, 500 | `[2,64,27]` | 통과 | 통과 | 0 |
| Raisins | 1006, 1506 | `[2,64,27]` | 통과 | 통과 | 0 |
| AlphabetSoup | 1991, 2491 | `[2,64,27]` | 통과 | 통과 | 0 |

처음 원격 검사기는 이미지 cache에 없는 `phase`를 읽어 CPU 검사 뒤 `KeyError`로 끝났다.
authoritative source NPZ에서 phase를 읽고 cache action 길이와 대조하도록 고쳤다. 이미 통과한
네 CPU 단계를 정확히 확인한 경우에만 DINO 단계부터 재개하게 했으며, 회귀 테스트를
추가했다. 이 과정에서 데이터·가중치·student 모델·teacher 출력은 바꾸지 않았다.

전체 대응, 해시, 전송 영수증과 원격 결과는
[`bundle-release.json`](bundle-release.json)에 고정했다. 이 묶음은 **모두 train인 진단
데이터**다. 학습된 정책의 HOPE 폐루프 성공이나 보지 못한 물체 일반화 증거가 아니다.

## G3. 다음 단일 요인 실험

다음 실험은 새 학습이 아니라, 기존 단일 병 x0 최종 정책의 출력에만 **teacher 시범에서
고정 산출한 관절별 한 tick 명령 변화 한계**를 적용하는 3회 폐루프 비교다. 모델, 데이터,
체크포인트, seed 1000–1002, 974 tick, 성공·품질 계약을 모두 고정한다. 원본 teacher의
974개 명령은 이 한계로 하나도 바뀌지 않는다.

실행서와 고정값은 [`plans/STUDENT_NEXT_SINGLE_FACTOR.md`](../../../plans/STUDENT_NEXT_SINGLE_FACTOR.md),
[`plans/STUDENT_NEXT_SLEW_LIMITS.json`](../../../../plans/STUDENT_NEXT_SLEW_LIMITS.json)에 있다.
평가기의 기본 경로는 바꾸지 않았고 `--slew-limits`를 명시할 때만 적용된다. **이 실험과
새 HOPE 학습은 실행하지 않았다.**

## 검사

- 이번 범위의 직접 회귀 테스트 51개 통과.
- Dexonomy 환경에서 전체 572개 테스트 통과. 기존 NumPy deprecation warning 138개만 남음.
- 새·수정 Python 파일 Ruff 통과.
