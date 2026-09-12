<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# Milk multiccd 소규모 teacher 수집

2026-09-11, **완료 (23:07)** — multiccd B에서 Milk teacher 시범 1건 accepted·동일 해시 재현
(`initial_gate_pass=true`, `sustained_stability_pass=false`, student-ready 아님). 목표: Milk에서 접근→닫기→리프트→유지를 통과하고 재현되는 시범 확보.
기본 물리 설정·코드, 기존 데이터·student 작업은 바꾸지 않는다. 새 학습 없음.

## 1. 기존 보고 정정

[multiccd 비교](../hope-multiccd-ab/README.md)의 "같은 시작 상태"를 **"물체의 시작 상태가 같다"**로 고쳤다.
로봇의 brace 명령은 B에서 달랐으므로 전체 시작 상태가 같다고 쓰지 않는다. 파지 결과 차이의 원인도
"손–물체 접촉 영향과 로봇 시작 명령 차이를 나눌 수 없다"로 고쳤다.

## 2. Milk 전용 물리 설정 (별도 실행 설정)

- B = MuJoCo `mjENBL_MULTICCD` 한 비트. 실행 범위 hook [dexonomy_multiccd/sitecustomize.py](../../../../scripts/dexonomy_multiccd/sitecustomize.py)가
  (a) Dexonomy `MuJoCo_BaseEnv`(정렬 matcher·정련·평가 환경)와 (b) 실행기 `icra.scene.build`가 컴파일한 모델에
  컴파일 직후 플래그를 켠다. 기존 테이블 hook은 같은 파일에서 원본 그대로 불러온다. 이 세션의 모든 단계와
  정련 worker는 `PYTHONPATH`로 이 hook을 받고, 모델마다 플래그 상태를 `multiccd-activation.jsonl`에 남긴다.
- 생성 전 preflight: 정련 환경(`op/gen_grasp_admm.py`와 같은 방식으로 만든 `MuJoCo_OptEnv`)과 실행기 모델의
  실제 multiccd·MuJoCo 버전·접촉 관련 설정을 기록하고, **둘 다 플래그가 켜져 있고** Milk가 B(자체 brace 포함,
  즉 수집 경로 그대로)에서 기존 1.2 s 판정을 통과할 때만 진행한다. 1.2 s·마지막 0.5 s 원시 수치를 남긴다.
- 정련기와 실행기는 원래부터 다른 물리다(정련기는 중력 끔·자체 cone·margin의 준정적 최적화). 이는 기존
  파이프라인 설계이며 바꾸지 않고 기록만 한다. 이번에 맞추는 불일치는 multiccd다.
- 공식 크기·질량·관성, 배치(중앙, yaw 0), 마찰, 제어, 품질·성공 기준은 기존 그대로.
- 대조군: [multiccd 비교](../hope-multiccd-ab/README.md)의 AlphabetSoup B 성공(episode `7e46fd28…`, model
  `d6d7417f…`, initial state `2979f645…`)을 이 hook 경로로 1회 재현. 세 해시가 모두 같아야 Milk 실행으로 간다.

## 3. 후보와 예산 (실행 전 고정)

- Milk 중앙·yaw 0 한 조건. 유형 `1_Large_Diameter`, `10_Power_Disk`(정식 init_tmpl, 해시 기록).
- align seed 0·1, 정련 400, matcher 4, 예측 100(기존 HOPE 설정). 관측·예측은 1회 저장 후 재사용.
- 선택: 내용 해시로 중복 제거 후 최대 8개. 유형×seed 네 칸(LD s0, LD s1, PD s0, PD s1)에서 해시 순으로
  한 칸씩 돌아가며 뽑고, 같은 파일은 한 번만(모든 parents 보존). 칸이 비면 남은 칸에서 계속.
- 최초 성공 최대 2건 각 1회 재현. 물리 실행 최대 **11회**(대조군 1 + 새 후보 8 + 재현 2). 인프라 실패·재시도도
  시뮬레이션을 시작하면 차감.
- 실패해도 유형·배치·scale·반복 수·제어·기준을 바꾸지 않는다.

## 결과

### preflight (22:08, exit 0, `preflight.json`)

| 항목 | 정련 환경 (`MuJoCo_OptEnv`) | 실행기 모델 |
| --- | --- | --- |
| MuJoCo | 3.3.3 | 3.3.3 (같은 venv) |
| multiccd | **켜짐** (enableflags 16) | **켜짐** (enableflags 16) |
| nativeccd | 켜짐 | 켜짐 |
| timestep / solver / iterations | 0.002 / Newton(2) / 100 | 0.002 / Newton(2) / 100 |
| ccd 반복 / 허용오차 | 50 / 1e-6 | 50 / 1e-6 |
| 중력 | **꺼짐** (정련기 설계) | 켜짐 |
| cone / impratio | pyramidal / 1 | elliptic / 10 |
| geom margin | 0, 0.001, 0.02 m | 기존 실행기 값 |
| 테이블 | scene_cfg에 `table` 항목 (Milk·table) | box geom |

multiccd 불일치는 없다. 중력·cone·impratio·margin 차이는 기존 파이프라인의 정련기 설계이며 바꾸지 않았다.

**Milk 안정화 — B, 자체 brace(수집 경로 그대로)** 원시 수치:

| 판정 | 값 | 기준 | 결과 |
| --- | --- | --- | --- |
| 1.2 s 이동 / 회전 | 0.740 mm / 0.581° | ≤ 5 mm / ≤ 5° | 통과 |
| 1.2 s 선속도 / 각속도 | 0.00207 m/s / 0.02304 rad/s | ≤ 0.005 / ≤ 0.05 | 통과 |
| 1.2 s 로봇–물체 / 로봇–테이블 침투 | 0 / 0 | ≤ 1 µm | 통과 |
| 1.2 s 목표 참조 (머리 / 손목) | 가능 / 불가 | 하나 이상 | 통과 |
| **기존 1.2 s 판정** | | | **통과** |
| 마지막 0.5 s 선속도 최대 / RMS | 0.00484 / 0.00116 m/s | ≤ 0.005 (최대) | 통과 |
| 마지막 0.5 s 각속도 최대 / RMS | **0.0981** / 0.0230 rad/s | ≤ 0.05 (최대) | **실패** |
| 마지막 0.5 s 물체–테이블 접촉점 평균 / 최소 | 3.74 / 1 | — | — |
| **마지막 0.5 s 지속 판정** | | | **실패** |

주의: 정지 A/B에서는 B를 A의 안정화 전 상태에서 시작해 Milk가 지속 판정도 통과했다(각속도 최대 0.0497).
수집 경로(B 자체 brace)에서는 1.2 s 수치는 같지만 마지막 0.5 s 각속도 최대가 0.098 rad/s로 지속 판정을
통과하지 못한다. 기존 계약은 1.2 s 판정이므로 사전 규칙대로 진행했지만, **Milk의 안정화 여유는 이 경로에서
지속 기준 밖**이라는 점을 결과 해석에 함께 둔다.

**상태 표기 (사용자 지시, 모든 결과·HANDOFF에 동일):** `initial_gate_pass = true`,
`sustained_stability_pass = false`, 조건 = multiccd B·**B 자체 brace 명령**(수집 경로)·Milk 중앙·yaw 0.
파지가 accepted여도 Milk를 "충분히 안정화된 물체"로 표현하지 않는다. HANDOFF의 모든 항목은
`student_ready = false`, `merge_into_existing_student_data = false`이며 기존 student 데이터에 자동 편입하지 않는다.
AlphabetSoup B 대조군이 세 해시로 재현되지 않으면 Milk 동적 실행은 시작하지 않는다(코드로 강제).

### 정적 생성 (22:09–22:55, 모든 단계 exit 0, timeout 0, 부분 출력 없음)

| 유형 | 예측 (요청 100) | 도달 가능 | seed | init 필터 입력 | loss 통과 | 충돌 통과 | QP 통과 | 패딩으로 되돌아온 수 (충돌/QP) | 정렬 후 유지 = init 파일 | 정련 | 선별 | 선별 탈락 (손이 이미 물체 안) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Large Diameter | 38 | 108 | 0 | 108 | 0 | 3 | 7 | 105 / 101 | 76 | 44 | 11 | 33 |
| | | | 1 | 108 | 0 | 4 | 9 | 104 / 99 | 75 | 44 | 11 | 33 |
| Power Disk | 17 | 42 | 0 | 42 | 0 | 0 | 3 | 42 / 39 | 30 | 14 | 6 | 8 |
| | | | 1 | 42 | 0 | 0 | 3 | 42 / 39 | 32 | 13 | 3 | 10 |

- GraspGenX는 요청 100개에 38·17개만 냈다(예측 1회, 저장본 재사용). 도달 가능 수가 예측보다 큰 것은 필터의
  roll·깊이 변형 때문이다.
- **init 필터를 실제로 통과한 수와 정렬 출력은 다르다.** loss는 모든 seed에서 0 통과이고, 정련에 들어간 파일은
  거의 전부 `valid_index_padding`으로 되돌아온 후보다. init 파일 수를 통과 수로 보고하지 않는다. 정련은 자체
  필터로 따로 판정한다.
- 선별 후 내용 해시 기준 **고유 후보 30개**(칸별 LD s0 11, LD s1 11, PD s0 6, PD s1 3).

**고정한 실행 순서 (8개, 칸별 차례·해시 순, 중복 1회)**

| 순서 | 유형 | 후보 | parents |
| ---: | --- | --- | --- |
| 0 | Large Diameter | `03b66901b0d9` | LD s0 |
| 1 | Large Diameter | `0d08f94b3c6c` | LD s1 |
| 2 | Power Disk | `082ccd42d360` | PD s0 |
| 3 | Power Disk | `130183329cf0` | PD s1 |
| 4 | Large Diameter | `21f559ab1cc3` | LD s0, LD s1 (같은 파일) |
| 5 | Large Diameter | `12a7ccf32792` | LD s1 |
| 6 | Power Disk | `1f36229cfa2e` | PD s0 |
| 7 | Power Disk | `3b754104a9e3` | PD s1 |

**multiccd 실제 적용 기록** (`multiccd-activation.jsonl`): 모델을 컴파일한 모든 프로세스에서 플래그 켜짐 —
정렬 4개(Dexonomy 환경·실행기 scene 모두), **정련 worker 8개(`dexrun`)**, 필터 2, 관측 1, 배치 검사 1, preflight.
선별(screening) 단계의 기록은 없다 — 선별이 MuJoCo 모델을 컴파일하지 않는 것으로 보이지만 **확인하지 않았다**.
기록에 보이는 `collect_teacher_dataset.py` 1개 프로세스는 선별이 아니라 동적 단계의 AlphabetSoup 대조군 실행이다.

### 동적 실행 (22:55–23:06, exit 0, 인프라 오류·timeout·재시도 0)

**대조군**: AlphabetSoup B 성공을 이 hook 경로로 재실행 → accepted, **episode·모델·초기 상태 해시 3개 모두 이전 B
실행(`7e46fd28…`)과 동일**. 보관 모델 multiccd 켜짐. 따라서 Milk 실행을 진행했다.

**Milk 새 후보 8개** (모두 B, 보관 모델 multiccd 켜짐 확인). 실제 종료 phase와 접촉 분류는 별도 열이다.

| 순서 | 유형 | 후보 | 결과 | 실제 종료 phase | 접촉 분류 | 시작 시 물체 선/각속도 (m/s, rad/s)·테이블 접촉점 | 동시 최대/종료 접촉군 | 최대 상승 | 획득 전 XY 이동 |
| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| 0 | LD | `03b66901b0` | `no_validated_home_path` | (시범 기록 전 접근 경로 검증 실패; `simulation_started=true`, 예산 차감) | — | — | — | — | — |
| 1 | LD | `0d08f94b3c` | closure_timeout | close | 모두 동시 성립 없음 | 0.0020 / 0.0308 · 4 | 4/2 | 5.5 mm | 24.5 mm |
| 2 | PD | `082ccd42d3` | closure_timeout | close | 모두 동시 성립 없음 | 0.0002 / 0.0119 · 4 | 4/0 | 38.6 mm | 58.3 mm |
| 3 | PD | `130183329c` | closure_timeout | close | 성립 후 상실 | 0.0002 / 0.0119 · 4 | 5/1 | 6.0 mm | 37.2 mm |
| 4 | LD | `21f559ab1c` | closure_timeout | close | 성립 후 상실 | 0.0020 / 0.0308 · 4 | 5/2 | 5.7 mm | 41.9 mm |
| 5 | LD | `12a7ccf327` | no_quality_feasible_command | close | 모두 동시 성립 없음 | 0.0020 / 0.0308 · 4 | 2/2 | 0.2 mm | 2.8 mm |
| 6 | PD | `1f36229cfa` | closure_timeout | close | 모두 동시 성립 없음 | 0.0002 / 0.0119 · 4 | 4/4 | 1.4 mm | 16.3 mm |
| **7** | **PD** | **`3b754104a9`** | **accepted** | **hold** | **completed** | 0.0002 / 0.0119 · 4 | **5/5** | **151.2 mm** | 5.0 mm |

(시작 상태는 후보의 열린 손 자세에 따라 두 가지로 나뉜다. 두 경우 모두 물체는 테이블 접촉점 4개로 서 있다.)

- **새 후보 승인 1/8.** 실패 7건: 시범 기록 전 접근 경로 검증 실패 1(시뮬레이터는 시작했고 초기 상태가
  보관돼 있으며 예산에서 차감됨 — "시뮬레이션 전 실패"가 아니다), close 종료 6(모두 동시 성립 없음 4, 성립 후 상실 2).
  lift에 들어간 실패는 없다. 여러 실패에서 0.5초 연속 접촉 전 물체가 17–58 mm 밀렸다(키 큰 물체가 닫는 중에
  밀리거나 흔들린 것으로 보이며, 원인 분리는 하지 않았다).
- **재현**: `3b754104a9` 1회 재실행 → accepted, **episode·모델·초기 상태 해시 모두 동일** (`10916e76…`,
  `8ad75c14…`, `ef1895ce…`). 두 번째 성공이 없어 두 번째 재현 슬롯은 쓰지 않았다.
- **예산**: 물리 실행 **10 / 11**(대조군 1, 새 후보 8, 재현 1). 입력 거절 0.

**승인 시범 — Milk 중앙, `10_Power_Disk` `3b754104a9`** (parents: Power Disk seed 1)

| 항목 | 값 |
| --- | --- |
| 물리 | multiccd B(기본값 아님), 보관 모델에 플래그 켜짐 |
| 안정성 | **`initial_gate_pass = true`, `sustained_stability_pass = false`** (B 자체 brace 조건) |
| episode / model / initial state | `10916e763a3f9554…` / `8ad75c1433f9bc48…` / `ef1895cecec6f375…` |
| 자산 메시 / 템플릿 | mesh `7f2ad636356e726c…`, Power Disk `218d1c0448a6…` (LD `08f364dfe141…`) |
| 품질 | 침투 1.866 mm(여유 0.134), 손가락 힘 13.3 N(여유 16.7), 테이블 침투 0, 접근 변위 0.15 mm, 상승 151.2 mm, 접촉 유지 상승 1.16 s, 접촉군 5/5 |
| 근접 한계 tick | close·lift·hold 모두 침투 ≥1.9 mm 0%, 힘 ≥29 N 0% |
| 획득 전 XY 이동 | 5.0 mm (student guard와 같은 판정 아님) |

## 결론

- **Milk에서 접근→닫기→리프트→유지를 통과하고 동일 해시로 재현되는 teacher 시범 1건을 multiccd B 조건에서
  확보했다.** 이것은 teacher accepted까지이며 **student-ready가 아니다.**
- 이 물체의 안정성 표기는 `initial_gate_pass = true`, `sustained_stability_pass = false`(B 자체 brace)다.
  파지가 성공했다고 Milk를 충분히 안정화된 물체로 보지 않는다.
- B는 기본 물리가 아니다. 이 시범을 기존 기본 물리 데이터·student 묶음과 섞지 않는다(`HANDOFF.json`에
  `merge_into_existing_student_data = false`). B 채택 여부는 [multiccd 비교](../hope-multiccd-ab/README.md)의
  채택 전 검증 범위가 정해진 뒤 결정할 일이다.
- 새 후보 수율 1/8, 실패 대부분이 close 단계에서 물체가 밀린 경우라, 형상이 달라진 뒤에도 접촉 획득이 병목이다.
  추가 튜닝은 하지 않았다.

## 보충 (23:22)

성공 시범의 실제 보관 초기 상태 진단은 [hope-milk-state-diagnosis](../hope-milk-state-diagnosis/README.md)와
`/data1/yspark/sim/icra/hope-milk-state-diagnosis-v1/SUPPLEMENT.json`에 따로 기록했다. 기록 명령 재생은 원본과 오차 0으로
재현됐고, 두 보관 시작 상태 모두 물체가 제자리에서 작게 흔들렸다(마지막 0.5 s 각속도 최대 0.058·0.084 rad/s). 아래
`initial_gate_pass`/`sustained_stability_pass` 값은 바꾸지 않았다.

## 인계와 상태

- **`/data1/yspark/sim/icra/hope-milk-multiccd-v1/HANDOFF.json`** (sha256 `bdb65b98abe868fe…`): 모든 기존 HANDOFF와
  분리. 상단과 항목마다 `initial_gate_pass`, `sustained_stability_pass`, 안정성 조건과 원시 수치, `student_ready=false`,
  `merge_into_existing_student_data=false`; 정련기·실행기 물리 설정과 multiccd 활성 기록; episode·모델·초기
  상태·scene·메시·템플릿 해시, 후보·parents, 재현 해시, 품질 여유.
- 기타: `preflight.json`, `preflight-milk-settle.npz`, `multiccd-activation.jsonl`, `static/`, `selected.json`,
  `plan.json`, `control.json`, `ledger.json`, `runs.json`, `static-table.json`, `chain-status.tsv`, 로그.
- 코드: [run_hope_milk_multiccd.py](../../../../scripts/experiments/run_hope_milk_multiccd.py),
  [dexonomy_multiccd/sitecustomize.py](../../../../scripts/dexonomy_multiccd/sitecustomize.py), 테스트
  `tests/test_dexonomy_multiccd_hook.py`(4개)·`tests/test_compare_multiccd.py`·`tests/test_hope_new_shapes.py` 통과,
  Ruff 통과. 기본 설정·공통 코드·기존 데이터·student 파일은 수정하지 않았다.
- 실행 22:08–23:07. 남은 프로세스·tmux 없음(확인).
