<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 기존 HOPE 풀에서 새 teacher 후보 검증

> **정정 (2026-09-11 20:55, [교차 확인](../hope-diversity-crosscheck/README.md) 반영).** 아래
> "닫기 5 / 운반 4"는 **실행 phase가 아니라 접촉 성립 여부에 따른 분류**다(`contacts_never_all_taken`
> 5, `contacts_taken_then_lost` 4). 저장된 phase를 다시 읽으면 새 중앙 실패 9건의 **실제 종료 phase는
> close 8건, lift 1건**(AlphabetSoup LD `0ebc6a25`만 lift에 진입)이다. "운반" 4건 중 3건
> (AlphabetSoup PD `0e60951a`, Peaches LD `e5e86be8`, Peaches PD `36db3f64`)은 **close 중에** 모든
> 접촉군을 한 번 성립시킨 뒤 잃은 것이며, **리프트 하중을 유지하지 못한 실패로 해석하면 안 된다.**
> 질문 4의 "닫기(5)·운반(4)" 서술도 같은 뜻으로 읽는다. 분류 코드와 teacher 판정은 바꾸지 않았다.

2026-09-11, **완료 18:10** (실행 약 27분, 차감 24/30). T0 17:35:49 KST, 새 후보·배치 시작 마감 21:35:49(T+4h),
강제 종료 23:35:49(T+6h). 실행서
[CLAUDE_TEACHER_HANDOFF_2026-09-11](../../../plans/CLAUDE_TEACHER_HANDOFF_2026-09-11.md).

새 GraspGenX·align·refine, 새 물체, 그릇·망치 튜닝, 제어·접촉군·품질·성공 기준 변경은
하지 않았다. 기존 HOPE 정련·선별 풀에서 **아직 실행하지 않은 후보만** 기존 제어 그대로
home 접근→닫기→리프트→hold 전체 경로로 실행한다. 새 형상 일반화 실험이 아니다.

- 출력: `/data1/yspark/sim/icra/hope-candidate-diversity-v1`
  (repo에는 링크 `out/hope-candidate-diversity-v1`만). 기존 `out/hope-pilot`의 출력·HANDOFF·
  모델은 읽기만 했다.
- 실행: turing, tmux `icra-div`, nice 15·BLAS 1스레드, 동시 수집 1개. GPU 사용 없음.
- 도구: [run_hope_candidate_diversity.py](../../../../scripts/experiments/run_hope_candidate_diversity.py)
  (`plan`/`run`/`report`). 단계 실행·전체 로그·프로세스 그룹 종료는 이미 검증한
  [run_hope_pilot.py](../../../../scripts/run_hope_pilot.py)의 `Journal`을 그대로 쓴다.

## C0. 첫 HOPE 실행의 실제 비용 정정

`prior-cost-ledger.json` — 첫 실행의 모든 수집(유효·격리)의 `attempts.json`과 단계 로그에서
시도마다 한 줄을 만들었다(attempt UUID, 물체/배치, 후보 hash, 단계·수집 경로, 단계 시작 시각,
초기 상태 저장 시각, 시뮬레이션 시작 여부, 종료 결과, episode hash, 인프라 오류, retry-of).

| 구분 | 수 | 설명 |
| --- | ---: | --- |
| 최종 유효 새 시도 | **48** | 중앙 36 + 배치 12. 성공률 분모 |
| 재현 | **8** | 추가 시범이 아니다 |
| 인프라 오류로 격리된 완결 episode | **4** | Raisins 첫 execute(디스크 가득 참). episode ID 0000–0003, 각 hash가 재실행본과 **동일** |
| 인프라 오류로 중단된 부분 시작 | **1** | Raisins teacher_0004: 로그에 `starting`, 초기 상태 파일 0 byte·모델 43 MB로 잘림. 결과 **unknown/partial** |
| 시뮬레이션 전 입력 검사 거절 | 12 | AlphabetSoup v2 execute(평면 자산 정체 검사 오류). 물리 0회 |
| **물리 시작 합계** | **61** | 48 + 8 + 4 + 1 |

48은 최종 유효 시도 수이며, 격리된 5건은 독립 실패로 분모에 다시 넣지 않지만 계산 비용과
재시도에서 빼지 않는다. 재실행된 Raisins 시도 5개는 ledger에 `retry_of`로 원래 시도와 연결된다.

## C1. 새 후보 고정 (실행 전)

`plan.json`에 규칙·예산·마감·코드 해시와 아래 파일들의 sha256을 고정했고, `run`은 이 해시가
바뀌면 실행을 거부한다.

- 풀: 물체별 유형·seed manifest를 `build_expansion_manifest.pooled()`로 합치고(모든 parents 보존)
  source 내용 sha256로 중복 제거. **source 파일 해시 불일치 0**.
- 제외(`excluded.json`): 첫 실행의 어느 수집(유효·격리, 성공·실패 무관)에든 들어간 후보.
  물체당 12개(배치·재현은 같은 성공 후보라 추가 없음).
- 선택(`selected.json`): 계획 유형 순서대로 유형별 hash 순 앞 2개, 부족한 유형은 다음 유형의
  남은 후보로 hash 순 보충, 유형 round-robin으로 실행.
- control(`control.json`): 기존 중앙 성공 3개. 새 후보 통계에 넣지 않는다.

| 물체 | 풀 (고유) | 제외 | 남은 후보 (유형별) | 실행 순서 (유형, source sha256 앞 10자) |
| --- | ---: | ---: | --- | --- |
| AlphabetSoup | 195 | 12 | PD 91 · LD 92 | PD `0e60951a58` → LD `0ebc6a2569` → PD `1443cca6ce` → LD `107c4737e2` |
| Peaches | 66 | 12 | PD 53 · LD **1** | PD `36b35b00b7` → LD `e5e86be8c5` → PD `36cebe9806` → PD `36db3f6472` |
| Raisins | 48 | 12 | PP **0** · LD 36 | LD `3e8fa79936` → LD `3f2973a68d` → LD `438146b501` → LD `47209fd29a` |

- 부족: Peaches Large Diameter 1개뿐(1 부족 → Power Disk로 보충), Raisins Palmar Pinch 0개
  (첫 실행에서 정련 통과가 1개였고 그것도 선택에서 제외됨 → 4개 모두 Large Diameter).
- AlphabetSoup LD `0ebc6a2569`는 seed 1·2 두 선별에 같은 파일로 있어 parents 둘을 보존하고
  한 번만 실행한다.
- 이 규칙은 첫 실행의 "전체 hash 상위 12"와 다르므로 첫 실행의 1/12와 공정한 수율 비교가 아니다.

예산(`plan.json`): control 3, 새 중앙 12, 첫 새 성공 재현 3, 배치(yaw +15°, X −20 mm; 사전 고정)
6, 배치 성공 재현 6, **합계 30**. 시뮬레이션을 시작한 실행은 인프라 오류·재시도라도 차감한다.
인프라 오류는 1회만 재시도(차감)한다. T+4h 이후 새 후보·배치 시작 금지, 재현은 T+6h까지.

## C2. 실행

### 기존 성공 control (물체당 1회)

기존 중앙 성공의 한 줄 manifest(첫 실행의 `*-centre-reproduce.json`, 해시 고정)를 같은 scene·
수집 인자로 data1에 다시 실행했다. **3/3 accepted, episode sha256이 원본과 동일.**
컴파일 모델·초기 상태 해시도 첫 실행과 같다. 실행 환경·코드 경로가 원본과 같다는 확인이므로
세 물체 모두 새 후보 실행으로 진행했다.

| 물체 | attempt | 시작 | episode sha256 (= 원본) | model | initial state | 소요 |
| --- | --- | --- | --- | --- | --- | ---: |
| AlphabetSoup | `d1dc49dd` | 17:43:21 | `2ad2b2647274105e…` | `16f0700373ed…` | `cdf7a3a3a944…` | 90.7 s |
| Peaches | `5d161a18` | 17:44:52 | `3e88a9afedb41312…` | `f711c25bfc5b…` | `538414e596ed…` | 62.1 s |
| Raisins | `703c27cd` | 17:45:54 | `fe233e9e22b1ffd0…` | `89121bacba2e…` | `96713c98c9b3…` | 57.0 s |

차감 3/30. control은 새 후보 통계와 추가 시범 수에 넣지 않는다.

### 새 중앙 후보 (물체당 4개, 고정 순서)

분류는 공통 `icra.outcomes`([summarise_hope_attempts.py](../../../../scripts/diagnostics/summarise_hope_attempts.py)):
닫기 = 필요한 접촉군 5개를 동시에 잡지 못함, 운반 = 모두 잡은 뒤 상실.

| 물체 | 순서 | 유형 | 후보 | 결과 | 멈춘 단계 | 동시 최대/종료 접촉군 | 최대 상승 |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| AlphabetSoup | 0 | PD | `0e60951a58` | closure_timeout | 운반 | 5/3 | 2.3 mm |
| AlphabetSoup | 1 | LD | `0ebc6a2569` | lift_timeout | 운반 | 5/4 | 23.1 mm |
| AlphabetSoup | 2 | PD | `1443cca6ce` | **accepted** | 완주 | 5/5 | 144.3 mm |
| AlphabetSoup | 3 | LD | `107c4737e2` | closure_timeout | 닫기 | 4/4 | 3.4 mm |
| Peaches | 0 | PD | `36b35b00b7` | no_quality_feasible_command | 닫기 | 3/3 | 0.8 mm |
| Peaches | 1 | LD | `e5e86be8c5` | closure_timeout | 운반 | 5/4 | 1.3 mm |
| Peaches | 2 | PD | `36cebe9806` | no_quality_feasible_command | 닫기 | 3/3 | 0.6 mm |
| Peaches | 3 | PD | `36db3f6472` | closure_timeout | 운반 | 5/0 | 4.5 mm |
| Raisins | 0 | LD | `3e8fa79936` | **accepted** | 완주 | 5/5 | 128.6 mm |
| Raisins | 1 | LD | `3f2973a68d` | closure_timeout | 닫기 | 4/3 | 3.5 mm |
| Raisins | 2 | LD | `438146b501` | **accepted** | 완주 | 5/5 | 143.6 mm |
| Raisins | 3 | LD | `47209fd29a` | no_quality_feasible_command | 닫기 | 3/2 | 4.4 mm |

**새 중앙 승인 3/12** — AlphabetSoup 1/4, Peaches 0/4, Raisins 2/4. 실패 9건은 닫기 5, 운반 4이며
접근 실패는 없었다. 실패 사유는 모두 `execution_terminated_early`, `incomplete_full_approach`,
`insufficient_contact_preserving_hold`. Peaches는 새 성공이 없어 규칙대로 재현·배치를 실행하지
않았다.

### 첫 새 성공 재현과 사전 고정 배치

| 물체 | 첫 새 성공 | 재현 | yaw +15° | 재현 | X −20 mm | 재현 |
| --- | --- | --- | --- | --- | --- | --- |
| AlphabetSoup | PD `1443cca6ce` | 동일 `e15f27809a66…` | **accepted** | 동일 `41d6c926176a…` | **accepted** | 동일 `c38792a50594…` |
| Raisins | LD `3e8fa79936` | 동일 `0c6b54becd4b…` | **accepted** | 동일 `06354bdd56f8…` | 실패 (lift_timeout) | — |
| Peaches | 없음 | — | 미실행 | — | 미실행 | — |

- 재현 5회 모두 episode·컴파일 모델·초기 상태 해시가 원본과 같고 둘 다 accepted.
- Raisins 두 번째 새 성공(`438146b501`)은 규칙상 재현·배치하지 않았다(첫 성공만). HANDOFF에
  `reproduction_note`로 표시했다.
- Raisins X −20 mm: 5/5 접촉을 766 tick 유지하며 102.5 mm까지 들었지만 lift/hold 시간 예산이
  끝날 때까지 품질 기준의 접촉 유지 hold를 채우지 못했다(`still_holding_when_the_budget_ran_out`,
  사유 `insufficient_contact_preserving_hold`). 기준 미달이며 다시 분류하지 않는다.
- 배치 상대좌표(`placement-transfer-<object>.json`): 새 중앙 성공과 네 배치 실행(실패 포함)
  전부 통과 — teacher 목표와 물체 frame 오차 ≤ 1.2e-16 m, 계획의 안정 자세와 0.00 mm,
  중앙 목표를 계획 변환으로 옮긴 것과 0.00 mm/0.00°, 테이블 위 높이 불변.

### 비용 (ledger)

| 역할 | 차감 / 상한 |
| --- | ---: |
| 기존 성공 control | 3 / 3 |
| 새 중앙 후보 | 12 / 12 |
| 첫 새 성공 재현 | 2 / 3 (Peaches 없음) |
| 새 후보 배치 | 4 / 6 (Peaches 미실행 2) |
| 배치 성공 재현 | 3 / 6 |
| **합계** | **24 / 30** |

인프라 오류 0, 재시도 0, 입력 검사 거절 0, 예산·마감·디스크로 미실행 0, timeout 0.
24회 모두 시뮬레이션을 시작했고 모두 차감했다. 실행 17:43:21–18:10:09(약 27분), T+4h 이전에 끝나
남은 시간과 예산은 쓰지 않았다(지시대로 채우지 않음).

## C3. 품질 여유

같은 정의로 새 실행 전부(`margins.json`)와 **첫 실행 승인 8건**(`old-accepted-margins.json`;
원본과 해시가 같은 한 줄 재현 수집에서 계산)을 측정했다. 여유와 accepted는 별개다.
근접 한계 비율(침투 ≥ 1.9 mm, 손가락 힘 ≥ 29 N인 tick 비율)은 서술용이며 새 판정 기준이 아니다.
"접촉 유지 상승" = 모든 요구 접촉군을 유지하면서 물체 최소 높이 신호가 0.1 m 이상인 가장 긴 연속 구간.
"획득 전 XY 이동" = 요구 접촉군을 0.5초 연속 유지하기 전까지의 물체 XY 이동 최대값(student guard와
비슷하지만 최소 힘 조건이 없어 **같은 판정은 아니다**).

| 승인 시범 | 후보 (유형) | 최대 침투 (여유) | 최대 힘 (여유) | 침투 ≥1.9 mm 비율 close/lift/hold | 힘 ≥29 N 비율 close | 접촉 유지 상승 | 획득 전 XY 이동 |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| **새** AlphabetSoup 중앙 | `1443cca6` PD | 2.000 mm (0.000) | 29.9 N (0.1) | 0.18 / 0.02 / 0.00 | 0.07 | 2.64 s | **1.08 mm** |
| **새** AlphabetSoup yaw +15° | 〃 | 1.985 mm (0.015) | 29.3 N (0.7) | 0.06 / 0.01 / 0.00 | 0.01 | 2.66 s | 1.12 mm |
| **새** AlphabetSoup X −20 mm | 〃 | 2.000 mm (0.000) | 29.9 N (0.1) | 0.11 / 0.00 / 0.00 | 0.02 | 2.64 s | 1.23 mm |
| 기존 AlphabetSoup 중앙 | `057c04e1` PD | 1.568 mm (0.432) | 14.7 N (15.3) | 0 / 0 / 0 | 0 | 1.00 s | **10.41 mm** |
| 기존 AlphabetSoup yaw −15°·X ±20 mm | 〃 | 1.43–1.74 mm (0.26–0.57) | 13.0–16.1 N | 0 / 0 / 0 | 0 | 1.00 s | 7.99–8.87 mm |
| **새** Raisins 중앙 | `3e8fa799` LD | **1.678 mm (0.322)** | 9.8 N (20.2) | 0 / 0 / 0 | 0 | 2.60 s | 0.88 mm |
| **새** Raisins yaw +15° | 〃 | **1.561 mm (0.439)** | 9.9 N (20.1) | 0 / 0 / 0 | 0 | 2.56 s | 0.79 mm |
| **새** Raisins 중앙 | `438146b5` LD | 1.999 mm (0.001) | 15.5 N (14.5) | 0.22 / 0.23 / **1.00** | 0 | 2.66 s | 1.04 mm |
| 기존 Raisins 중앙 | `0ddf3447` LD | 2.000 mm (0.000) | 11.9 N (18.1) | 0.06 / 0.15 / 0.00 | 0 | 2.80 s | 2.29 mm |
| 기존 Raisins yaw +15° | 〃 | 1.995 mm (0.005) | 11.3 N (18.7) | 0.07 / 0.42 / 0.00 | 0 | 2.80 s | 1.51 mm |
| 기존 Peaches 중앙·X +20 mm | `203b95f1` PD | 1.961–1.999 mm (0.001–0.039) | 28.9 N (1.1) | ≤0.06 / 0 / 0 | 0 | 2.78–2.80 s | 1.96–2.46 mm |

테이블 침투는 모든 승인 시범에서 0, 접근 변위는 0이다. 기존 AlphabetSoup의 "접촉 유지 상승 1.00 s"는
그 후보가 hold 1초를 채운 직후 실행이 끝났기 때문이며, 상승 중 유지 시간을 다른 후보와 길이로
비교하지 않는다. 손가락 수·요구 접촉군이 유형마다 달라 힘 합계를 우열로 보지 않았고, 여유들을
하나의 점수로 합치지 않았다.

### 보고 질문에 대한 답

1. **새 고유 teacher 후보의 완주가 생겼는가?** 그렇다. 새 고유 후보 **3개**(AlphabetSoup PD
   `1443cca6ce`, Raisins LD `3e8fa79936`·`438146b501`)가 중앙에서 accepted, 그중 순서상 첫 성공 2개는
   동일 재현했다. Peaches는 4개 모두 실패. 새 teacher 시범은 **6건**(중앙 3 + 배치 3), 물체 2·메시 2·
   물체×배치 조건 5(AlphabetSoup 중앙·yaw +15°·X −20 mm, Raisins 중앙·yaw +15°)·후보 3·유형 2.
2. **기존보다 여유가 나은 사례가 있는가?** Raisins `3e8fa799`는 기존 Raisins 후보보다 침투 여유가
   분명히 크고(0.322·0.439 mm 대 0.000·0.005 mm) 근접 한계 tick이 없으며 힘 여유도 비슷하거나 크다.
   AlphabetSoup `1443cca6`은 **trade-off**다 — 침투·힘 여유는 거의 0으로 기존(0.43 mm·15.3 N)보다
   나쁘지만, 0.5초 연속 접촉 전 물체 이동이 1.1 mm로 기존(10.4 mm)보다 훨씬 작다. 후자는 기존
   AlphabetSoup 중앙이 student 인수에서 보류된 원인과 같은 종류의 양이다. Raisins `438146b5`는 hold
   전 구간이 1.9 mm 이상으로 여유가 얇다.
3. **같은 두 배치에서의 차이.** AlphabetSoup: 기존 후보는 yaw +15° 실패(닫기)·X −20 mm 성공,
   새 후보는 둘 다 성공. Raisins: 기존·새 후보 모두 yaw +15° 성공·X −20 mm 실패(둘 다 접촉을 유지한
   채 lift/hold 시간 예산에서 기준 미달). 조건마다 n = 1이라 배치 강건성이 후보에 달린 것인지는
   **미결정**이다. 일반화 성공률로 쓰지 않는다.
4. **새 형상 확대의 근거인가, 접촉 유지가 병목인가.** 기존 풀의 다른 후보에서 두 물체는 새 완주를
   얻어 후보 다양성은 늘릴 수 있음을 보였지만, 새 중앙 실패 9건이 모두 닫기(5)·운반(4)이고
   접근 실패는 0이다. Peaches는 첫 실행과 합쳐 16개 후보 중 1개만 완주했고 실패 후보 다수가 힘·침투
   한계 근처였다(`36db3f64` 침투 2.000 mm·힘 29.9 N). **접촉 획득·유지가 여전히 병목**이다. 이번 결과는
   형상을 늘리기 전에 이 병목의 원인 분리가 필요하다는 쪽의 근거이며, 형상 확대 자체를 반대하는
   근거는 아니다. 어느 쪽도 이 실행으로 결정하지 않았다.

## 인계

**`/data1/yspark/sim/icra/hope-candidate-diversity-v1/HANDOFF.json`** — 원본
`out/hope-pilot/HANDOFF.json`과 분리. 승인 6건마다 episode·모델·초기 상태 경로와 sha256, scene·메시·
템플릿 해시, 후보 sha256·유형·parents(유형·seed 선별 출처), 재현 해시(또는 규칙상 미재현 사유),
이번 세션 control 대응(원본 재현 여부), 품질 여유 전체.

- Teacher accepted는 student ready가 아니다. **Student 획득 계약 검사는 GPT-5.6에 인계**하며,
  GPT의 고정 7건 묶음에 새 자료를 직접 추가하지 않았다.
- 참고: 새 AlphabetSoup 후보의 획득 전 XY 이동(1.1–1.2 mm)은 기존 AlphabetSoup 보류 사례(10.4 mm)와
  다르지만, 위 정의는 student guard와 같지 않으므로 실제 guard 검사로만 판단해야 한다.
- 여유가 거의 없는 승인: 새 AlphabetSoup 3건(침투 2.000 mm·힘 29.9 N까지), 새 Raisins `438146b5`.

기타 산출물: `ledger.json`(시도별 원장), `margins.json`, `old-accepted-margins.json`,
`attempts-summary.json`, `placement-transfer-<object>.json`, `prior-cost-ledger.json`,
`plan.json`·`candidate-pool.json`·`excluded.json`·`selected.json`·`control.json`, 단계별 로그 `logs/`,
실행 journal `stages.json`, 콘솔 `run.log`.

## 코드와 검사

- 신규 [run_hope_candidate_diversity.py](../../../../scripts/experiments/run_hope_candidate_diversity.py)와
  [tests/test_hope_candidate_diversity.py](../../../../tests/experiments/test_hope_candidate_diversity.py)(선택 규칙
  4개·예산/마감 4개). 기존 `run_hope_pilot.py`의 Journal·격리·모델 공유를 그대로 사용했다.
- 실행 후 `report`에만 `reproduction_note` 필드를 추가했다(보고 전용; 실행 경로·plan 파일 해시 불변).
- 실제로 실행한 검사: `tests/test_hope_candidate_diversity.py`, `tests/test_hope_pilot_wrapper.py`,
  `tests/test_teacher_scene_identity.py` 27개 통과; 신규 두 파일 Ruff 통과. 전체 suite는 실행하지 않았다.
- 기존 control 3/3이 원본 episode 해시를 재현했다(위 표).
- 산출 해시: `HANDOFF.json` `22e6c645…`, `ledger.json` `c13269dc…`, `margins.json` `b765852c…`.

## 종료 상태와 재개

- 18:10 종료. tmux `icra-div` 종료, 이 작업의 프로세스 없음(확인). 다른 사용자 작업은 건드리지 않았다.
- 출력 1.4 GB(`/data1` 여유 291 GB). 같은 scene의 동일 모델은 hash 확인 후 hard link로 공유.
- 재실행/재보고: 계획은 고정돼 다시 쓰지 않는다. 보고만 다시 만들려면
  `third_party/dexonomy/.venv/bin/python scripts/run_hope_candidate_diversity.py report --out /data1/yspark/sim/icra/hope-candidate-diversity-v1`.
  `run`은 ledger를 읽어 완료된 시도를 건너뛰며, 남은 예산 6회는 이번 규칙상 쓸 대상이 없다.
- 공통 `execution.py`·`scene.py`·`dataset.py`·Recorder, student/visual/mask/goal-memory, 기존
  `out/hope-pilot` 산출물·HANDOFF, `PAPER_TODO.md`·`RESUME.md`·날짜별 index는 수정하지 않았다.
