# 전체 논문 TODO — 두 단계 학습

갱신 2026-09-12. [연구 정의](RESEARCH_PLAN.md)를 기준으로 한다.
[이전 진행판](records/plans/archive/2026-09-12/PAPER_TODO.md)은 당시 기록이다.
최신 실측은 [당일 최종 판단](records/experiments/2026-09-12/day-feasibility/DECISION.md)을 따른다.
새 direct/residual BC 학습·평가는 완료했으며 물리·품질 기준은 변경하지 않았다.

## 현재 위치

**① 학습된 파지 선정·시범 적응·자동 수집과 ② 목표 조건부 시각 행동 학습이 모두 미완성이다.**
정적 후보·동적 시범 확보, 학습 코드 실행, 최종 과제 성공을 구분한다.
전체 진행률이나 논문 성공 확률을 하나의 숫자로 환산하지 않는다.

| ID | 일 | 확인된 상태 | 완료 조건 |
| --- | --- | --- | --- |
| P0 | 두 단계 목표·입력·평가 정의 | 사용자 합의 반영 | 형상 분할·예산·목표 선택 과제 확정 |
| P1 | Dexonomy 후보·실행·기록 기반 | 최근 3물체·6후보 배치 49/72 성공, 성공49건 재현 | 형상/배치/후보/접촉·물리 계약 추적 |
| P2 | 형상 기반 유형·후보 선정 학습 | 소규모 학습·평가 완료; logit 정정 후 Raisins 5/8은 같은 풀의 사후 비교 | 미지 형상에서 기준 순위 대비 실행 수율·비용 평가 |
| P3 | Dexonomy 시범 적응 RL·자동 수집 | 학습 미검증; 고정 손목편집24회 개선0, 접근 PPO와 별개 | 같은 후보/예산의 무수정 대비 개선, 전체 실패 분모 |
| P4 | 시각 데이터 인수 | 기존 A12 통합 완료; 후속49건은 정확한 중복5·신규44, 별도 시각 인수 중 | 해시·재생·입력·cache·모델 검사, 충분한 형상 범위 |
| P5 | 작은 시각 BC 전체 실행 | 단일 병 x0·low-noise·direct·residual·공식 행동 backend 최종 각각0/3 | 카메라+목표+관절로 home→리프트·유지 |
| P6 | Mask/Point/Language 물체 선택 | mask 경로 있음; 전체 지시 행동 미검증 | 동일 장면 목표 변경 성공, grounding 실패 분리 |
| P7 | 미지 형상·배치·가벼운 clutter | teacher의 제한된 시범 사례; student 미검증 | 형상 분리 평가·여러 seed·실패 분해 |
| P8 | Dexonomy 선정·적응의 기여 | 미검증 | 동일 student/data/compute 비교 및 수집 비용 |
| P9 | 실제 로봇 | 최종 시스템 근거 미확보 | 보정/실데이터 사용 공개, 목표 파지 평가 |
| P10 | 원고·재현 자료 | 초안 목표 개정 | 모든 주장에 근거·비교·한계·재현 경로 |

ID는 이번 개정에서 재정의했다. 이전 문서의 P2/P3/P4는 역사적 분류이며 위 표와 번호를
직접 대응시키지 않는다. 부위 mask/부위 파지는 필수 완료 항목에서 제외했다.

## 근거의 경계

- [접근 BC](records/experiments/2026-09-09/approach-bc-v1/README.md)와
  [PPO 비교](records/experiments/2026-09-09/kl-and-seeds/README.md)는 고정 후보의 privileged 팔 접근 실험이다.
  1번 전체 RL이 불가능하거나 실패했다고 결론낼 근거가 아니다.
- [HOPE 인수](records/experiments/2026-09-11/hope-diversity-student-intake/README.md)는 기존7+새5 인수 기록이다.
  [12개 통합 작업](records/experiments/2026-09-11/hope-twelve-student/README.md)의 전체 완료는 원격 release로 확인한다.
- [최종 회수](records/experiments/2026-09-12/lecun-final-check/README.md): 기존 두 BC 최종0/3과 A12 통합 완료 확인.
- [새 BC 비교](records/experiments/2026-09-12/direct-bc-feasibility/README.md): direct/residual 각6,331갱신 후0/3.
  기록 명령 및 같은 궤적의 비시각 최근접 재사용은 성공. 학습된 시각 파지 성공은 아니다.
- [시작 상태 범위](records/experiments/2026-09-12/day-feasibility/RESET_SCOPE.md): 유형별 open에서 시작하므로
  공통 중립 home에서 유형 선택을 학습했다고 주장할 수 없다.
- [Milk 초기상태 검토](records/experiments/2026-09-11/milk-state-crosscheck/README.md): B teacher 성공·재현은 유지,
  지속 안정화 미통과·학생 미인수. A12에 포함하지 않는다.
- [Claude clutter 보고](records/experiments/2026-09-11/hope-clutter/README.md)는 완료를 보고했으며
  Raisins 방해물1/2조건 성공을 적고 있다. 이는 teacher 사례이고 학습된 선정/student의 증거가 아니다.
  이번 문서 작업에서 원시 결과를 독립 재검증한 것으로 표현하지 않는다.

## 다음 작업

이전 [24시간 계획](records/plans/SPRINT_24H_2026-09-12.md)은 종료됐다.
현재 상태는 [RESUME](RESUME.md), 다음 작업의 기준은 [CURRENT_WORK](plans/CURRENT_WORK.md)에서 확인한다.
아래 비교 원칙은 유지하되 과거 완료 job을 재시작하지 않는다.

1. [최종 판단](records/experiments/2026-09-12/day-feasibility/DECISION.md)에 따라 현재 student의 대규모 확대를 보류한다.
   검증된 행동 학습 구현과 최소 재현 비교를 다음 작업으로 삼는다. 완료 job은 재실행하지 않는다.
2. 물체 형상·배치·후보와 단계별 실행 결과를 연결한 선정 학습 계약 및 train/test 분할을 만든다.
3. Dexonomy 선정 모델과 DemoFunGrasp식 편집 RL을 각각 제한된 비교로 검증한다.
4. 검증 시범을 시각 BC로 연결하고 동일 장면 물체 선택 반사실 평가를 수행한다.
5. 데이터 규모·미지 형상·clutter·hardware·기여 실험으로 확대한다.

단일 병 시각 BC 진단을 무기한 전체 연구의 선행 조건으로 두지 않는다.
1번 학습 설계/자동 수집과 2번 진단을 분리 진행하되, 현재 모델·원본·물리 기준은 보존한다.
