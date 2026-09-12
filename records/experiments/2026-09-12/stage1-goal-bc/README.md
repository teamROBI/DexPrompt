# 1단계 목표 조건부 BC (BC→PPO 전 단계) — 학습 조건 완주 실패, PPO 미실행

2026-09-12 22:44–23:40. 목적: 정확한 물체·장면 형상, 로봇 상태, Dexonomy 파지 목표를 입력으로 팔·손 행동을 내는 정책이
**목표가 주어진 상태에서** teacher 시범을 실행할 수 있는지 확인(유형·목표 자동 선정은 범위 밖). Mask·language·시각 모델 없음.

**결론: BC는 학습 조건 6개에서 완주 0/6(행동 정의 두 가지 모두). 규칙대로 PPO는 시작하지 않았다.**
데이터·실행 연결은 정상(teacher 명령은 같은 평가 경로에서 6/6 재현), 개루프 적합도 양호하지만 폐루프에서 7–35 tick 안에 이탈한다.

## 1. 자료와 실행 위치

- 학습 자료: stage-1 데이터 계약 기본 풀의 HOPE 75 episode(A 물리, 단일 물체, 현재 controller), 78,445 프레임.
- 패키지: turing `/data1/yspark/sim/icra/stage1-goal-bc-v1/package` → lecun `/data1/yspark/projects/dexprompt/stage1-goal-bc-v1/package`
  (모델 37개 7.3 GB, 초기 상태, 영상 없는 배열; 양쪽 `MANIFEST.sha256` 일치). 형식: [`FORMAT.md`](FORMAT.md) (DP3 공용).
- 코드: 새 모듈 `scripts/stage1_bc/`(build_package, archive, replay, obs, policy, plan, train_bc, evaluate, diagnose),
  테스트 `tests/test_stage1_bc.py`(5개 통과). 공용 실행기·물리·성공 기준·기존 데이터·모델은 수정하지 않았다.
- lecun 고정 사본(쓰기 금지, `CODE.sha256` 양쪽 일치, import가 사본을 가리킴 확인): `code-v1`(재생) · `code-v2` · `code-v3`(설정·학습·평가) ·
  `code-v4`(+진단) · `code-v5`(절대 행동 정의). `src/icra` 7개 파일은 모든 사본에서 동일. 해석기: Codex venv(torch 2.6, MuJoCo 3.3.3, 읽기 전용).
- 자원: lecun GPU 2(학습 각 ~1분), CPU worker 3–4. GPU 0·1(다른 사용자)·turing 물리 실행은 쓰지 않았다.

## 2. 원본 재현 (재생) — 수치 차이와 판정을 따로

| 항목 | 결과 |
| --- | --- |
| 재생 수치 차이 (lecun, 75개, 기록된 최종 명령을 수집기와 같은 순서로) | `right_state`와 신호 10종 **모두 비트 단위 동일**(최대 차이 0). turing 1개도 동일 |
| 원본 teacher 판정 재현 (수집기의 phase 기반 `assess_episode`, turing에서 원본 배열로 재계산) | 75/75 accepted, 원본과 일치 |
| phase 없는 계약 `contact_acquisition_guard_v2` (student 평가 계약, 판정 재현과 별개) | 72/75 통과. 실패 3개는 모두 AlphabetSoup `057c04e1` 중앙·x−10 mm·yaw−7.5°의 `approach_displacement`(teacher '닫기' 구간의 물체 밀림을 원본 판정은 검사하지 않음) |

- 행동 직전의 미저장 상태(물체 자세·속도, 관절 속도, 손 루트 자세, 팔 보상 항)는 재생 중 `runs/replay-all/augment/`에 원본과 분리해 기록했다.

## 3. 공통 정의 (BC·PPO 공용, 실행 전 `config.json`에 기록)

- 관측(176차원): 관절 위치·속도, 물체 자세·속도, 손 루트 자세, 목표 손 자세와 오차, 손가락 열림·닫힘 목표와 오차, 직전 구간 접촉군·힘·상승량,
  경과 시간, 물체 크기 서술자. teacher phase·결과·미래 상태는 넣지 않음.
- 행동: 27개 최종 actuator 명령을 그대로 넣음(clip만, **중력 보상 재적용 없음**). v1 `명령 = q + u·척도`, v2 `명령 = 평균 + u·표준편차`.
- 정책: 정규화 MLP 3×512 Gaussian actor(PPO용 critic 구조 공유). BC는 MSE, 6000 step, 마지막 체크포인트.
- 개발 평가(학습 조건, 결과 보기 전 규칙으로 선택: 물체별 teacher가 계약 v2를 통과한 episode 중 이름순 첫 중앙·첫 비중앙 6개):
  결정적 평균 행동, 품질 위반 또는 0.5 s 유지 성공 또는 teacher 길이+250 tick에서 종료, 계약 v2 판정. 최종 비교 조건은 Codex와 별도 고정(여기서 정하지 않음).

## 4. 결과

| 실행 | 학습 손실 | 학습 조건 완주 | 첫 이탈 | 실패 형태 |
| --- | ---: | ---: | --- | --- |
| bc-v1 (상대 명령) | 0.0080 | **0/6** | 10–23 tick, 주로 손가락 | 테이블 충돌 4, 손가락 힘 1, 접근 중 물체 밀림 1 |
| bc-v2 (절대 명령, 이것만 변경) | 0.00013 | **0/6** | 6–8 tick | 침투 2, 테이블 충돌 2, 손가락 힘 1, 시간 초과 1 |

진단(`diagnose.json`, 학습 조건 6개):
- 데이터·실행 연결: 기록된 teacher 명령을 평가 경로(관측→행동 변환→clip→진행→판정)로 넣으면 6/6 성공(행동 변환의 부동소수점 반올림으로 비트 동일은 아님, 판정은 일치).
- 개루프: teacher 관측에서 BC 명령 오차 평균 v1 0.0004 rad, v2 팔 0.002·손가락 0.001 rad(초반 60 tick 최대 v2 팔 0.017 rad).
- 폐루프: 명령 오차가 v1은 tick 35에 0.3–0.7 rad(손가락 명령이 열린 자세에서 멀어짐), v2는 tick 20–30에 0.4–1.3 rad.
- 민감도(`analysis/sensitivity.json`, 순전파만): 관절 위치 0.005 rad 교란에 대한 명령 증폭률 v1 ≈ 1.0(명령 ≈ q + 상수 → 적분기), v2 ≈ 0.5(1 미만).
  경과 시간 1 tick에는 거의 무반응. v2의 폐루프 발산은 관절 위치·시간만으로는 설명되지 않는다 — 관절 속도·손 자세·목표 오차 등 다른 관측을 통한
  경로이거나 학습 분포 밖 상태에서의 일반적 누적 오차일 수 있으나 **확인하지 않았다**.

## 5. 미실행·다음

- PPO: BC가 학습 조건을 완주하지 못해 **실행하지 않았다**(규칙). critic 준비·KL 감시·퇴행 중단 설계는 obs/policy 모듈에 구조만 공유돼 있다.
- 다음 선택지(각각 새 실험 결정이라 실행하지 않음): (a) 관측 한 요소씩 제거(관절 속도·손 자세·목표 오차), (b) teacher가 교란 상태에서
  다시 계산한 복귀 시범(Codex 계획의 가설과 겹침), (c) 행동 묶음·시간 평활, (d) teacher 재계산 기반 DAgger류 수집.
- 결과 파일 사본: [`lecun-results/`](lecun-results/) (`SHA256SUMS`). 원본 실행 폴더는 lecun `runs/` (쓰기 금지).
