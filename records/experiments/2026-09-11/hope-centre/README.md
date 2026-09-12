<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# HOPE 중앙 3물체 teacher 전체 실행

2026-09-11, **완료 05:59** (전체 실행 약 3시간 21분, 새 시도 예산 48/48 소진). T0 02:38 KST. 새 생성 시작 마감 09:38(T+7h), 강제 종료 11:38(T+9h).
기준 문서 [TEN_HOUR_SPLIT_2026-09-11](../../../plans/TEN_HOUR_SPLIT_2026-09-11.md),
입력 [hope-pilot](../hope-pilot/README.md)·`out/hope-pilot-v1/generation-plan.json`.

## 정정

직전 대화 보고의 두 내용을 바로잡는다.

1. **예측 수 111 → 140.** 보고한 "111 proposed → 111 clear"는 driver 실행 전에
   손으로 돌린 smoke test(02:06:35)의 수치다. driver가 02:07:24에 예측을 **다시**
   실행해 140개를 냈고, 그 뒤 필터(`poses 140`)·정렬·정련은 모두 140개 쪽을 썼다.
   111개 파일은 같은 경로에 덮어써져 남아 있지 않다. GraspGenX 예측이 seed로 고정되지
   않아 재실행하면 다른 후보 집합이 나온다는 뜻이기도 하므로, 이후 wrapper는 완료된
   예측을 다시 실행하지 않는다.
2. **그릇 정련 0 → 과거 10개 있음.** "그릇·망치는 모든 설정에서 정련 0"은 틀렸다.
   [penetration-and-scale](../../2026-09-10/penetration-and-scale/README.md)의
   scale 0.08 × matcher 4 조건에서 그릇은 정련 **10개**(seed별 1/6/3)가 나왔고,
   정적 검사 10개 통과·동적 실행 10회·**완주 0**이었다. 정련 0은 scale 0.05와
   matcher 25 조건들에 한한 결과다.

## 실행 위치

- 서버 **turing**(lecun 아님). 기존 계획대로 teacher 환경이 준비된 곳에서 계속했고
  환경 이전은 하지 않았다. 네 GPU 모두 다른 사용자(jokim)의 작업이 99% 사용 중이며,
  예측만 **GPU 3**에서 짧게(약 13초, 1–2 GB) 실행했다. 다른 프로세스는 건드리지 않았다.
- CPU: nice 15, BLAS/OMP/MKL 1스레드, 정련 `n_worker=2`. tmux `icra-hope`.

## wrapper 보완 (v1 → v2)

v1([run_hope_pilot.py](../../../../scripts/run_hope_pilot.py) 첫 판)은 단계별 종료 코드와
출력 끝부분만 남겼고, 실패 뒤 차단이 없었으며, 정련 worker가 부모와 같은 프로세스
그룹이라 timeout 시 worker가 남을 수 있었다. v2는 다음 안전한 단계 경계에서 교체했다.

- 단계마다 **전체 출력 로그 파일**, 명령, cwd, 시작 시각, 종료 코드, timeout 여부, 로그 해시.
- 단계를 **자기 프로세스 그룹**으로 실행. timeout 시 그룹 전체 TERM→KILL, 종료 후
  그룹에 남은 프로세스를 KILL하고 PID를 기록. 신호는 자신이 띄운 그룹에만 보낸다.
- placement·observe 실패 → 그 물체 차단. predict·filter 실패 → 그 유형 차단.
  align 실패/timeout → 그 seed의 정련 차단. refine 실패/timeout → `grasp_data`
  부분 출력을 격리 디렉터리로 **이동**(삭제 아님)하고 select·pool에서 제외.
  select 실패 → 반쯤 쓴 manifest를 격리. manifest 종료 코드 3은 `no_candidates`로 분리.
- 재개 시 journal에서 완료된 단계를 건너뜀. 예측은 절대 재실행하지 않음.
- 생성 예산(예측 100·matcher 4·정련 400·seed 0/1/2·물체당 12·총 48+15)을
  `run-config.json`에 고정하고 값이 다르면 실행 거부.
- 09:38 이후 새 생성·새 동적 시도 시작 금지, 11:38에 모든 단계 강제 종료 상한.

## 실행 계약 오류와 수정 (v2 → v3, 03:19)

AlphabetSoup의 첫 execute는 **12개 후보 전부가 시뮬레이션 전에 거부**됐다
(`teacher source scene identifies a different target object`). 물리 실패가 아니다.

- 원인: `collect_teacher_dataset.checked_candidate`의 teacher 정체 검사가 DGN의
  경로 형식 `<set>/scene_cfg/<object>/<floating|tabletop>/scaleNNN.npy`만 가정하고
  `parents[1]`을 물체 이름으로 읽었다. HOPE(평면 자산)의 teacher는
  `…/objects/AlphabetSoup/scene_cfg.npy`를 가리키므로 `parents[1]`이 `objects`가 된다.
  teacher 파일과 scene 파일(`scene_id`, `task.obj_name`)은 모두 AlphabetSoup이다.
- 수정: `teacher_scene_object()`로 두 형식을 모두 읽도록 했다. 파일명이
  `scene_cfg.npy`면 부모 디렉터리, 아니면 기존대로 두 단계 위. 다른 물체의
  teacher는 여전히 거부한다. 품질·성공 기준, 접촉군, 제어는 바꾸지 않았다.
  회귀 검사 `tests/test_teacher_scene_identity.py` 5개 추가, 실제 AlphabetSoup
  후보 **12/12가 입력 검사 통과**를 확인했다.
- **v2 wrapper의 오분류도 정정한다.** v2는 수집 프로그램의 종료 코드 0만 보고
  AlphabetSoup을 "executed, 0/12 accepted"로 기록했다. v3는 시뮬레이션에 도달한
  시도가 0개인 수집을 `candidate_check_error`로 분류하고, 완료로 보지 않으며,
  동적 예산에서 세지 않는다. 오류난 수집 디렉터리는 격리(이동)해 보존하고 같은
  manifest(내용 해시 순 상위 12, 선택 규칙 불변)로 다시 실행한다. 이 12개는
  물리를 한 번도 돌리지 않았으므로 중복 실행이 아니다.
- Codex 인계: `scripts/collect_teacher_dataset.py`는 teacher 실행 스크립트이며 수집
  schema는 바꾸지 않았다. 각 episode의 `sources/`에 수정된 사본과 해시가 함께 남는다.
  v2 wrapper가 쓴 `handoff-centre.json`(0/12)은 무효이며 v3 실행 후 다시 쓴다.

## 교체 기록과 격리

| 시각 | 교체 | 중단한 단계 | 남은 프로세스 | 격리 디렉터리 (`out/hope-pilot/quarantine/`) |
| --- | --- | --- | --- | --- |
| 02:47 | v1 → v2 | AlphabetSoup Large Diameter filter (1분 경과) | 없음 | `20260911T024751`: 그 filter의 빈 출력 디렉터리 |
| 03:27 | v2 → v3 | Peaches Power Disk seed 1 align | 없음(그룹 비어 있음 확인) | `20260911T034136`: 그 seed 실험 디렉터리(빈 `new_tmpl`만) |
| — | v3 재실행 | AlphabetSoup execute(시뮬레이션 0회) | — | `20260911T032726`: 거부된 12개 수집 기록(15개 파일, episode 없음) |
| 04:15 | v3 → v4 | Raisins Palmar Pinch filter (1분 경과, 예측은 완료·보존) | 없음 | filter 부분 출력 (v4가 재실행 전 격리) |

**v3의 예산 이중 계산 (v4에서 수정).** v3는 journal 기록마다 `collect_out` 디렉터리의
시뮬레이션 시도를 더했다. 거부된 v2 execute 기록과 v3 재실행 기록이 같은
`AlphabetSoup/executed`를 가리켜 AlphabetSoup 12회가 두 번 세어졌다(실제 24회를 36회로
계산). 그대로면 Raisins 12회 뒤 48/48로 보여 배치 확대가 전부 막혔을 것이다. v4는
디렉터리당 한 번만 센다(회귀 검사 추가). 같은 교체에서 align의 빈 결과(exit 1 +
"0 aligned initializations")를 `empty_result`로 따로 기록하게 했다. v3 journal은
`stages-v3.json`(sha256 `ff56922c…`)으로 보존.

**v4 재개 시 빈 정렬 2건 중복 실행.** v4는 exit 0만 완료로 보았기 때문에 Peaches Large
Diameter seed 0·1의 빈 정렬(exit 1)을 다시 실행했다(각 약 35초). 원본은
`quarantine/20260911T041529`, `…041603`로 이동해 보존했고, 다시 낸 결과는 원본과
**행 66/66, kept 0/0, 충돌 목록·손 자세까지 동일**했다(seed 고정 정렬). 결과는 바뀌지
않았지만 지시한 "완료 단계 재실행 금지"에 어긋나므로 기록한다. 이후 코드는 빈 정렬을
완료로 보도록 고쳤다(`is_empty_alignment`, 회귀 검사 추가; 실행 중인 v4에는 영향 없음).

**디스크 가득 참 (05:15–05:22, v4 중단).** 루트 파일시스템(937 GB, `/home`·`/tmp` 공용)이
100%가 됐다. Raisins execute가 05:15에 시작해 05:18:40에 `attempts.json`을 쓰다
`ENOSPC`로 죽었고(5번째 모델 파일은 206 MB 대신 43 MB로 잘림), v4도 journal을 쓰다
`ENOSPC`로 종료했다. journal은 임시 파일→rename 방식이라 05:14의 77개 기록이 온전했다.
Raisins의 정적 단계는 모두 그 전에 끝났고 manifest 9개·정련 파일 74개가 모두 정상 로드된다.

- 원인: 최근 2시간 동안 `/home`·`/tmp`·`/var`에 쓰인 100 MB 이상 파일은 **이 파이프라인의
  컴파일 모델뿐**이었다. 텍스처가 모델에 들어가 시도당 약 209 MB이고 물체당 약 2.6 GB다.
  이미 거의 찬 공용 디스크를 이 실행이 넘겼다.
- 조치: (1) 내 세션 임시 디렉터리의 오래된 공개 저장소 clone 3개(graspxl·cyclo_lab·
  dexgrasp_zero, 2.8 GB)를 삭제. 다른 사용자·Codex 파일(`/tmp/icra-student-uv` 등)은
  건드리지 않았다. (2) 물체마다 13개 모델이 **바이트 단위로 같은 파일**이었으므로 hard link로
  저장만 공유했다. 원자적 rename으로 바꿔 경로가 사라지는 순간이 없고, 31개 경로 전부 해시를
  다시 계산해 **변경 없음**을 확인했다. 여유 공간 2.9 → 8.1 GB.
- v5 wrapper: 수집 단계마다 끝난 뒤 같은 방식으로 모델을 공유하고, 여유 공간이 3 GB 미만이면
  수집을 시작하지 않고 `not_run: disk space`로 기록한다(도중에 잘린 기록을 만들지 않기 위해).
- Raisins의 중단된 수집(시뮬레이션 약 4회 진행 후 중단)은 환경 오류이며 결과·예산에 넣지
  않는다. 격리 후 같은 manifest로 12회를 다시 실행한다(물리 재생은 결정론적).
- v5 재개(05:26)에서 빈 정렬을 완료로 보게 되자, Peaches Large Diameter seed 0·1에 대해
  **초기화 0개인 정련을 두 번 실행**했다(각 3.7·3.6초, 출력 없음, 선택 없음). 결과에 영향은
  없지만 불필요한 단계였다. 이후 코드는 빈 정렬 뒤 정련을 시작하지 않는다(회귀 검사 추가;
  실행 중인 v5에는 적용되지 않으며, 남은 빈 정렬이 없어 다시 일어나지 않는다).

교체는 정련(dexrun) 중이 아닐 때만 했고, 신호는 wrapper가 띄운 PID와 그 자식의 프로세스
그룹에만 보냈다. 교체 직전 journal은 `stages-v1.json`(sha256 `699f2bc7…`)과
`stages-v2.json`(`4d9a1061…`)으로, 콘솔 로그는 `run-v1.log`·`run-v2.log`로 보존했다.
v1 단계 기록에는 출력 끝부분만 있고 전체 로그 파일이 없다(정련은 Dexonomy 자체
`log/grasp_admm_0/main.log`가 남아 있다).

## 결과

### AlphabetSoup — 정적 단계 (중앙)

예측 수는 GraspGenX가 요청 100개에 대해 실제로 낸 수다. init 필터 값은 Dexonomy
필터가 **padding 전에** 통과시킨 수이고, "init 파일"은 padding으로 채워 넣은 뒤
실제로 쓰인 파일 수다. 둘은 다르다.

| 유형 | 예측 | 도달 가능 | seed | init 입력 | loss 통과 | 충돌 통과 | QP 통과 | init 파일 | 정련 | 선택 | 선택 탈락 (손이 이미 물체 안) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Power Disk | 140 | 102 | 0 | 102 | 0 | 2 | 72 | 69 | 54 | 39 | 15 |
| | | | 1 | 102 | 0 | 2 | 73 | 68 | 56 | 35 | 21 |
| | | | 2 | 102 | 0 | 1 | 75 | 70 | 51 | 33 | 18 |
| Large Diameter | 114 | 69 | 0 | 69 | 0 | 14 | 62 | 44 | 35 | 31 | 4 |
| | | | 1 | 69 | 0 | 15 | 66 | 47 | 40 | 34 | 6 |
| | | | 2 | 69 | 0 | 11 | 60 | 48 | 45 | 40 | 5 |

- **init 필터는 사실상 아무것도 통과시키지 않았다.** loss 필터 0/102·0/69, 충돌
  1–15개. 정련에 들어간 파일은 거의 전부 padding으로 되돌아온 후보다. 이는 matcher
  4회 설정에서 예상된 결과이며, init 통과 수로 보고하지 않는다. 정련은 자기
  필터(충돌·접촉·QP)로 따로 판정하므로 정련 통과 수는 유효하다.
- 테이블 훅: 정련 입력의 `scene_path`는 `out/hope-pilot-v1/objects/AlphabetSoup/scene_cfg.npy`
  (물체+table 두 항목)이며, 원본 loader의 단일 항목 assertion 없이 정련이 끝났다 —
  훅이 적용된 경우에만 가능한 결과다.
- 정련 합계 281, 선택 합계 212, **내용 해시 기준 고유 후보 195개**(17개는 두 seed
  이상에서 같은 teacher 파일). 해시 순 상위 12개: Power Disk 7·Large Diameter 5,
  seed 출처 PD s0 2·s1 3·s2 2, LD s0 2·s1 2·s2 1. 유형/seed 균형 배분이 아니다.

### AlphabetSoup — 동적 실행 (중앙, 12회)

home 자세에서 접근→닫기→리프트→hold 전체 실행, 기존 접촉 피드백 닫기·품질 기준
그대로. 실행 12회, 762.7초, exit 0, timeout 없음. **승인 1/12.**
분류는 [summarise_hope_attempts.py](../../../../scripts/diagnostics/summarise_hope_attempts.py)
(공통 `icra.outcomes` 이름 사용): 접근 = 검증된 home 경로 없음,
닫기 = 필요한 접촉군 5개를 동시에 잡지 못함, 운반 = 모두 잡은 뒤 리프트/hold 중 상실.

| episode | 유형 | 멈춘 단계 | 동시 최대 접촉군 | 전 접촉 tick | 최대 상승 | hold | 힘 여유 | 침투 여유 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_0000 | Power Disk | 접근 | — | — | — | — | — | — |
| teacher_0001 | Power Disk | 닫기 | 3/5 | 0 | 3.1 mm | 0 | 0.7 N | 0.01 mm |
| teacher_0002 | Power Disk | 운반 | 5/5 | 169 | 78.1 mm | 0 | 10.0 N | 0.32 mm |
| teacher_0003 | Power Disk | 닫기 | 0/5 | 0 | 0.0 mm | 0 | 30.0 N | 2.00 mm |
| teacher_0004 | Power Disk | 접근 | — | — | — | — | — | — |
| teacher_0005 | Power Disk | 닫기 | 4/5 | 0 | 0.9 mm | 0 | 10.7 N | 0.00 mm |
| teacher_0006 | Large Diameter | 닫기 | 4/5 | 0 | 0.6 mm | 0 | 8.5 N | 0.06 mm |
| **teacher_0007** | **Power Disk** | **완주** | **5/5** | **558** | **118.1 mm** | **1.00 s** | **15.3 N** | **0.43 mm** |
| teacher_0008 | Large Diameter | 운반 | 5/5 | 14 | 4.5 mm | 0 | 3.6 N | 0.03 mm |
| teacher_0009 | Large Diameter | 운반 | 5/5 | 103 | 32.8 mm | 0 | 0.2 N | 0.18 mm |
| teacher_0010 | Large Diameter | 닫기 | 0/5 | 0 | 0.0 mm | 0 | 29.9 N | 2.00 mm |
| teacher_0011 | Large Diameter | 닫기 | 4/5 | 0 | 0.8 mm | 0 | 1.0 N | 0.01 mm |

단계별: 접근 2, 닫기 6, 운반 3, 완주 1. 유형별: Power Disk 7회 중 1 완주,
Large Diameter 5회 중 0.

**승인 시범** — teacher_0007 (`10_Power_Disk`)

| 항목 | 값 |
| --- | --- |
| 후보 | `057c04e1d5755d4a…`, 출처 `hopeAlphabetSoup_10_Power_Disk_s0/…/0_41_grasp.npy` (PD seed 0 단독) |
| episode | `out/hope-pilot/AlphabetSoup/executed/episodes/teacher_0007_yaw_000.npz`, sha256 `2ad2b2647274105e…` |
| 모델·초기 상태 | model `16f0700373ed43f1…`, initial state `cdf7a3a3a9440861…` (mjSTATE_INTEGRATION) |
| scene | `checks-v3/AlphabetSoup/centre/scene.yaml`, sha256 `ab716a577219f175…` = 계획 해시 |
| 품질 | 침투 1.57 mm(한계 2), 손가락 힘 14.7 N(한계 30), 테이블 침투 0, 접근 변위 0, 상승 118.1 mm(최소 100), 접촉 유지 hold 1.00 s(최소 0.5), 접촉군 5/5 |

**재현 (1회, 동일 조건)** — 같은 manifest 행을 한 줄 manifest로 잘라 같은 scene·
수집 인자로 다시 실행했다(87초, exit 0).

| 대조 항목 | 결과 |
| --- | --- |
| episode sha256 | `2ad2b2647274105e…` = 원본, **동일** |
| 승인·상태 | 둘 다 accepted, `completed` |
| 컴파일 모델 해시 | 동일 |
| 초기 상태 해시 | `cdf7a3a3a9440861…` 동일 |
| 품질 판정 전체 | 모든 항목 동일 |
| 수집 코드 해시(`sources/`)·world 목표 자세 | 동일 |
| tick 수 | 1192 = 1192 |

`reproduce-centre/comparison.json`. 따라서 AlphabetSoup은 **배치 확대 조건을 만족**한다
(고정 후보 `057c04e1…`로 yaw ±15°, X ±20 mm 각 1회; 세 물체 중앙이 끝난 뒤 실행).

인계 파일: [handoff-centre.json](../../../../out/hope-pilot/AlphabetSoup/handoff-centre.json)
(accepted/rejected 분리, 모델·초기상태·후보 귀속 포함),
[attempts-summary.json](../../../../out/hope-pilot/AlphabetSoup/attempts-summary.json).

### Peaches — 정적 단계 (중앙)

| 유형 | 예측 | 도달 가능 | seed | init 입력 | loss 통과 | 충돌 통과 | QP 통과 | init 파일 | 정련 | 선택 | 선택 탈락 (손이 이미 물체 안) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Power Disk | 103 | 78 | 0 | 78 | 0 | 1 | 53 | 36 | 32 | 25 | 7 |
| | | | 1 | 78 | 0 | 4 | 50 | 35 | 29 | 23 | 6 |
| | | | 2 | 78 | 0 | 1 | 51 | 34 | 30 | 24 | 6 |
| Large Diameter | 131 | 66 | 0 | 66 | 0 | 0 | 56 | **0** | — | — | — |
| | | | 1 | 66 | 0 | 0 | 57 | **0** | — | — | — |
| | | | 2 | 66 | 0 | 0 | 55 | 2 | 1 | 1 | 0 |

정련 합계 Power Disk 91 + Large Diameter 1 = 92, 선택 73, **내용 해시 기준 고유 후보 66개**.
해시 순 상위 12개는 **전부 Power Disk**(seed 0: 5, 1: 3, 2: 6)이고, Large Diameter의 유일한
후보는 상위 12에 들지 않았다. 선택 규칙은 유형 균형이 아닌 내용 해시 순이므로 규칙대로
두었고, 이 결과를 유형별 성공률 비교로 쓰지 않는다.

### Peaches — 동적 실행 (중앙, 12회)

실행 12회(모두 Power Disk), 677.9초, exit 0, timeout 없음, 12회 모두 시뮬레이션 도달.
**승인 1/12.**

| episode | 멈춘 단계 | 동시 최대 접촉군 | 전 접촉 tick | 최대 상승 | hold | 힘 여유 | 침투 여유 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_0000 | 운반 | 5/5 | 31 | 1.1 mm | 0 | 3.6 N | 0.00 mm |
| teacher_0001 | 닫기 | 0/5 | 0 | 0.0 mm | 0 | 23.3 N | 1.67 mm |
| teacher_0002 | 닫기 | 3/5 | 0 | 0.4 mm | 0 | 4.9 N | 0.03 mm |
| teacher_0003 | 닫기 | 2/5 | 0 | 0.1 mm | 0 | 14.7 N | 0.01 mm |
| teacher_0004 | 닫기 | 2/5 | 0 | 0.7 mm | 0 | 1.4 N | 0.01 mm |
| teacher_0005 | 운반 (종료 시 5/5) | 5/5 | 20 | 1.8 mm | 0 | 0.9 N | 0.03 mm |
| teacher_0006 | 운반 | 5/5 | 21 | 2.7 mm | 0 | 5.7 N | 0.00 mm |
| teacher_0007 | 운반 | 5/5 | 8 | 4.3 mm | 0 | 0.1 N | 0.00 mm |
| **teacher_0008** | **완주** | **5/5** | **375** | **148.1 mm** | **1.00 s** | **1.1 N** | **0.04 mm** |
| teacher_0009 | 운반 | 5/5 | 21 | 1.1 mm | 0 | 4.0 N | 0.00 mm |
| teacher_0010 | 운반 | 5/5 | 13 | 2.5 mm | 0 | 0.0 N | 0.01 mm |
| teacher_0011 | 운반 | 5/5 | 4 | 1.6 mm | 0 | 0.1 N | 0.00 mm |

단계별: 닫기 4, 운반 7, 완주 1. AlphabetSoup과 달리 운반 실패 7건은 모두 5 mm 이하
상승에서 접촉을 잃었고, 여러 건이 힘·침투 한계에 거의 붙어 있었다(힘 여유 0.0–0.1 N,
침투 여유 0.00 mm). 접촉을 잡는 순간 이미 압력 완화 한계 근처라는 뜻이며, 원인
분리는 하지 않았다.

**승인 시범** — teacher_0008 (`10_Power_Disk`)

| 항목 | 값 |
| --- | --- |
| 후보 | `203b95f140d07f8a…`, 출처 `hopePeaches_10_Power_Disk_s2/…/0_10_grasp.npy` (PD seed 2 단독) |
| episode | `out/hope-pilot/Peaches/executed/episodes/teacher_0008_yaw_000.npz`, sha256 `3e88a9afedb41312…` |
| 모델·초기 상태 | model `f711c25bfc5b949c…`, initial state `538414e596edd36a…` |
| scene | `checks-v3/Peaches/centre/scene.yaml`, sha256 `ab716a577219f175…` = 계획 해시 |
| 품질 | 침투 **1.96 mm**(한계 2), 손가락 힘 **28.9 N**(한계 30), 테이블 침투 0, 접근 변위 0, 상승 148.1 mm, hold 1.00 s, 접촉군 5/5 |

**여유가 얇다.** 기준을 모두 만족해 승인됐지만 힘은 한계의 96%, 침투는 98%다.
student 데이터로 쓸 때 이 점을 표시해 인계한다. 기준은 바꾸지 않았다.

**재현 (1회, 동일 조건)** — 55.9초, exit 0. episode sha256 `3e88a9afedb41312…` **동일**,
둘 다 accepted·`completed`, 컴파일 모델·초기 상태·수집 코드 해시·world 목표·품질 판정
전체 동일, tick 1006 = 1006. `reproduce-centre/comparison.json`.
Peaches도 **배치 확대 조건을 만족**한다(고정 후보 `203b95f1…`).

인계 파일: [handoff-centre.json](../../../../out/hope-pilot/Peaches/handoff-centre.json),
[attempts-summary.json](../../../../out/hope-pilot/Peaches/attempts-summary.json).

### Raisins — 정적 단계 (중앙)

| 유형 | 예측 | 도달 가능 | seed | init 입력 | loss 통과 | 충돌 통과 | QP 통과 | 정렬 후 유지(init 파일) | 정련 | 선택 | 선택 탈락 (손이 이미 물체 안) | 정련 시간 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Palmar Pinch | 99 | 29 | 0 | 29 | 0 | 5 | 4 | 20 | **0** | — | — | 156 s |
| | | | 1 | 29 | 0 | 4 | 4 | 20 | 1 | **0** | 1 | 155 s |
| | | | 2 | 29 | 0 | 5 | 4 | 20 | **0** | — | — | 154 s |
| Large Diameter | 100 | 104 | 0 | 104 | 0 | 10 | 59 | 79 | 24 | 18 | 6 | 807 s |
| | | | 1 | 104 | 0 | 7 | 57 | 82 | 24 | 16 | 8 | 834 s |
| | | | 2 | 104 | 0 | 7 | 56 | 82 | 25 | 18 | 7 | 835 s |

(Large Diameter의 도달 가능 104 > 예측 100: 필터가 roll/깊이 변형을 만들기 때문. 정렬은
그중 100개를 사용.)

- **Palmar Pinch는 정련에서 사실상 전부 탈락했다.** 정렬 후 20개씩 남았으나 정련 통과는
  세 seed 합 1개였고, 그 1개도 선택 단계에서 "손이 이미 물체 안"으로 제외됐다. init QP 통과도
  29개 중 4개로, 계획에 둔 "형상 기반 가설"이 이 물체에서는 맞지 않았다. 새 유형은 추가하지
  않았다(지시대로).
- Large Diameter는 정련 73, 선택 52, **내용 해시 기준 고유 후보 48개**. 해시 순 상위 12개는
  전부 Large Diameter이며 seed 출처는 s0·s1·s2 각 5(합 15 — 3개가 두 seed에서 같은 파일).
- Raisins의 정적 단계는 모두 디스크가 차기 전(05:14:58)에 끝났고, 파일 무결성을 확인했다.

### Raisins — 동적 실행 (중앙, 12회)

v5에서 같은 manifest(sha256 `2af534ee…`, 디스크 오류로 중단된 실행과 동일 확인)로 12회 실행,
572.5초, exit 0, timeout 없음, 12회 모두 시뮬레이션 도달. **승인 1/12.**

| episode | 멈춘 단계 | 동시 최대 접촉군 | 전 접촉 tick | 최대 상승 | hold | 힘 여유 | 침투 여유 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_0000 | 운반 | 5/5 | 22 | 0.5 mm | 0 | 15.7 N | 0.02 mm |
| teacher_0001 | 닫기 | 4/5 | 0 | 9.8 mm | 0 | 7.6 N | 0.00 mm |
| teacher_0002 | 운반 | 5/5 | 109 | 9.7 mm | 0 | 14.3 N | 0.07 mm |
| teacher_0003 | 운반 (종료 시 5/5) | 5/5 | 1 | 0.5 mm | 0 | 15.3 N | 0.14 mm |
| **teacher_0004** | **완주** | **5/5** | **377** | **133.5 mm** | **1.00 s** | **18.1 N** | **0.0003 mm** |
| teacher_0005 | 닫기 | 3/5 | 0 | 5.8 mm | 0 | 17.1 N | 0.62 mm |
| teacher_0006 | 운반 (종료 시 5/5) | 5/5 | 253 | 104.2 mm | 0 | 14.3 N | 0.00 mm |
| teacher_0007 | 닫기 | 3/5 | 0 | 8.5 mm | 0 | 7.4 N | 0.70 mm |
| teacher_0008 | 닫기 | 4/5 | 0 | 2.7 mm | 0 | 6.9 N | 0.12 mm |
| teacher_0009 | 운반 | 5/5 | 3 | 7.2 mm | 0 | 6.5 N | 0.01 mm |
| teacher_0010 | 닫기 | 4/5 | 0 | 0.1 mm | 0 | 16.2 N | 0.69 mm |
| teacher_0011 | 닫기 | 4/5 | 0 | 0.4 mm | 0 | 11.4 N | 0.00 mm |

단계별: 닫기 6, 운반 5, 완주 1. teacher_0006은 104.2 mm까지 들고 종료 시에도 5/5였지만
`no_quality_feasible_command`로 조기 종료(902 frame)했고 품질 기준의 접촉 유지 hold가 0초라
거부됐다(사유 `execution_terminated_early`, `incomplete_full_approach`,
`insufficient_contact_preserving_hold`). 닫기 실패 중 여러 건이 5–10 mm 상승을 보인 것은 일부
손가락만으로 물체를 끌어올린 것으로 보이며, 요구 접촉군을 채우지 못했으므로 실패다.

**승인 시범** — teacher_0004 (`1_Large_Diameter`)

| 항목 | 값 |
| --- | --- |
| 후보 | `0ddf3447c061b986…`, 출처 `hopeRaisins_1_Large_Diameter_s1/…/0_56_grasp.npy` (LD seed 1 단독) |
| episode | `out/hope-pilot/Raisins/executed/episodes/teacher_0004_yaw_000.npz`, sha256 `fe233e9e22b1ffd0…` |
| 모델·초기 상태 | model `89121bacba2e3304…`, initial state `96713c98c9b3c533…` |
| scene | `checks-v3/Raisins/centre/scene.yaml`, sha256 `ab716a577219f175…` = 계획 해시 |
| 품질 | 침투 **1.99970 mm**(한계 2), 손가락 힘 11.9 N(한계 30), 테이블 침투 0, 접근 변위 0, 상승 133.5 mm, hold 1.00 s, 접촉군 5/5 |

**침투 여유가 0.0003 mm다.** 기준을 만족해 승인됐지만 사실상 한계선 위다. 1.5 mm 압력 완화
기준을 넘어 한계 직전까지 간 뒤 유지된 경우이므로, student 데이터로 쓸 때 이 점을 함께
인계한다. 기준은 바꾸지 않았다.

**재현 (1회, 동일 조건)** — 54.5초, exit 0. episode sha256 `fe233e9e22b1ffd0…` **동일**,
둘 다 accepted·`completed`, 컴파일 모델·초기 상태·수집 코드 해시·world 목표·품질 판정
전체 동일, tick 985 = 985. `reproduce-centre/comparison.json`.
Raisins도 **배치 확대 조건을 만족**한다(고정 후보 `0ddf3447…`).

인계 파일: [handoff-centre.json](../../../../out/hope-pilot/Raisins/handoff-centre.json),
[attempts-summary.json](../../../../out/hope-pilot/Raisins/attempts-summary.json).

### 중앙 조건 요약

| 물체 | 실행 유형 | 동적 시도 | 승인 | 재현 | 멈춘 단계 (접근/닫기/운반) | 승인 시범의 최소 여유 |
| --- | --- | ---: | ---: | --- | --- | --- |
| AlphabetSoup | PD 7 · LD 5 | 12 | 1 | 동일 | 2 / 6 / 3 | 침투 0.43 mm |
| Peaches | PD 12 | 12 | 1 | 동일 | 0 / 4 / 7 | 침투 0.04 mm, 힘 1.1 N |
| Raisins | LD 12 | 12 | 1 | 동일 | 0 / 6 / 5 | 침투 0.0003 mm |
| 합계 | | **36** | **3** | **3/3 동일** | 2 / 16 / 15 | |

세 물체 모두 중앙에서 재현 가능한 전체 시범 1개씩을 얻었다(teacher 기준). 계획의 "형상 기반
유형 가설" 중 실제로 승인에 이른 것은 AlphabetSoup·Peaches의 Power Disk와 Raisins의 Large
Diameter다. 다만 선택이 내용 해시 순이라 유형별 시도 수가 균형이 아니므로 유형 간 성공률
비교로 쓰지 않는다. 승인 3건 중 2건은 품질 한계에 매우 가깝다.

### 배치 확대 (중앙에서 재현한 3물체 × 4배치, 각 1회)

각 물체의 **중앙 첫 성공 후보를 고정**하고(AlphabetSoup `057c04e1…`, Peaches `203b95f1…`,
Raisins `0ddf3447…`) 계획의 배치 scene(yaw ±15°, X ±20 mm; YAML 해시 8/8 계획과 일치)에서
home 접근부터 전체를 1회씩 실행했다. scene에 배치가 들어 있으므로 episode yaw는 0(회전
중복 없음). 제어·마찰·접촉군·성공 기준은 그대로이며, 실패한 배치를 조정하거나 다시
시도하지 않았다. 성공한 배치는 같은 조건으로 1회 재현했다.

| 물체 | 배치 | 결과 | 상태 | 최대 상승 | hold | 힘 여유 | 침투 여유 | 재현 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| AlphabetSoup | yaw −15° | **완주** | completed | 145.7 mm | 1.00 s | 17.0 N | 0.57 mm | 동일 `54973e51…` |
| AlphabetSoup | yaw +15° | 닫기 실패 (0/5) | no_quality_feasible_command | 0.0 mm | 0 | — | — | — |
| AlphabetSoup | X −20 mm | **완주** | completed | 118.2 mm | 1.00 s | 13.9 N | 0.26 mm | 동일 `49df6b4a…` |
| AlphabetSoup | X +20 mm | **완주** | completed | 139.9 mm | 1.00 s | 14.6 N | 0.31 mm | 동일 `d6f637c3…` |
| Peaches | yaw −15° | 닫기 실패 (4/5) | no_quality_feasible_command | 0.8 mm | 0 | 0.3 N | 0.02 mm | — |
| Peaches | yaw +15° | 운반 실패 | closure_timeout | 2.2 mm | 0 | 0.7 N | 0.04 mm | — |
| Peaches | X −20 mm | 운반 실패 (근접) | lift_timeout | 149.7 mm | 0.50 s | 0.8 N | 0.03 mm | — |
| Peaches | X +20 mm | **완주** | completed | 148.7 mm | 1.00 s | 1.1 N | 0.0012 mm | 동일 `ebcec15f…` |
| Raisins | yaw −15° | 들고 있는 채 시간 초과 | lift_timeout | 46.6 mm | 0 | 18.5 N | 0.00 mm | — |
| Raisins | yaw +15° | **완주** | completed | 148.5 mm | 1.00 s | 18.7 N | 0.005 mm | 동일 `4089eeb6…` |
| Raisins | X −20 mm | 들고 있는 채 시간 초과 | lift_timeout | 79.7 mm | 0 | 18.2 N | 0.00 mm | — |
| Raisins | X +20 mm | 운반 실패 (종료 시 5/5) | closure_timeout | 0.8 mm | 0 | 15.8 N | 0.00 mm | — |

**배치 승인 5/12** (AlphabetSoup 3/4, Peaches 1/4, Raisins 1/4), 성공 5건 모두 재현
episode sha256 동일.

**teacher·물체·테이블 상대좌표 확인** ([check_placement_transfer.py](../../../../scripts/ops/check_placement_transfer.py),
결과 `out/hope-pilot/<object>/placement-transfer.json`). 성공·실패와 무관하게 15개 episode
전부(중앙 재현본 3 + 배치 12)에서 저장된 모델·초기 상태를 복원해 확인했고 **15/15 통과**:

- teacher 손 root 목표 = 안정된 물체 frame × teacher의 물체 기준 root: 오차 ≤ 1.6e-16 m, ≤ 3.7e-8 rad.
- 물체의 안정 자세 = 계획(Codex preflight)의 settled transform: 차이 0.00 mm / 0.00°.
- 배치의 손 목표 = 중앙 목표를 계획의 배치 변환으로 옮긴 것: 0.00 mm / 0.00°.
- 테이블 위 높이 불변: 물체 AlphabetSoup +0.2 · Peaches −0.1 · Raisins 0.0 mm,
  손 root +134.4 · +126.0 · +116.2 mm가 모든 배치에서 같다.

즉 **실패한 배치는 좌표 전달 오류가 아니라 같은 teacher 자세로 닫기·운반하는 물리의
실패**다. 품질 여유가 얇았던 Peaches·Raisins가 배치 변화에 더 약했고(각 1/4), 여유가
있던 AlphabetSoup이 3/4였다. 이는 관찰이며 원인 분리는 하지 않았다. Peaches X −20 mm는
149.7 mm까지 들고 접촉 유지 0.50 s를 채운 뒤 잃었고, Raisins yaw −15°·X −20 mm는 접촉을
모두 유지한 채 lift 시간 예산(1450 frame)이 끝났다(46.6·79.7 mm) — 셋 다 기준 미달이다.

## 최종 집계 (분모 포함)

| 항목 | 값 |
| --- | --- |
| 새 동적 시도 (시뮬레이션 도달) | **48 / 48** — 중앙 36 (물체당 12) + 배치 12 |
| 재현 실행 | **8 / 15** — 중앙 3 + 배치 5, **8/8 episode sha256 동일** |
| teacher 승인 시범 | **8** — 중앙 3 (3/36) + 배치 5 (5/12) |
| 고유 물체 / 물체×배치 / 고유 teacher 후보 | **3 / 8 / 3** |
| 단계 timeout | 0 |
| 예산·마감·디스크 부족으로 미실행 | 0 |

- **고유 teacher 후보는 3개뿐이다.** 배치 확대는 설계대로 물체마다 같은 후보를 다른 배치에서
  실행한 것이므로, 배치 성공 5건을 독립 teacher 후보로 세지 않는다. 같은 물체의 새 yaw·X는
  새 형상이 아니다.
- **분모에서 뺀 실행 (환경 오류, 결과 아님):** (1) v2의 AlphabetSoup execute — 12개 후보
  전부 입력 검사에서 거부, 시뮬레이션 0회(검사 코드 수정 후 재실행). (2) v4의 Raisins
  execute — 디스크 가득 참으로 약 4회 진행 후 중단, journal에 기록되지 못함(격리 후 재실행).
- **단계별 종료** (`HANDOFF.json`의 `stage_endings`): placement 3, observe 3, predict 6,
  filter 6 모두 정상; align 정상 16·빈 결과 4(Peaches LD seed 0·1 원본 2 + v4 중복 2, 결과 동일);
  refine 정상 16·정렬이 빈 seed에서 실행된 무출력 2; select 14; manifest 5(재구성 포함, 내용 동일);
  execute 정상 3·시뮬레이션 전 거부 1; expand 12; reproduce 8. 실패·timeout 0.
- 새 생성 마감(09:38)과 강제 종료(11:38) 전에 모두 끝났다. 7시간 이후 시작한 단계는 없다.

## 인계

**`out/hope-pilot/HANDOFF.json`** ([bundle_hope_handoff.py](../../../../scripts/data/bundle_hope_handoff.py)) —
승인 8건마다 episode·모델·초기 상태 경로와 sha256, scene·메시·템플릿 해시, 후보 ID와 유형·seed
귀속, 품질 판정 전체, 재현 비교 경로. 조건별 `out/hope-pilot/<object>/handoff-<placement>.json`에
accepted/rejected가 분리돼 있고 rejected는 episode 또는 diagnostics 경로를 가진다.
단계 journal `stages.json`, 단계별 전체 로그 `logs/`, 고정 설정 `run-config.json`.

student 쪽에서 쓸 때 알아야 할 점:

1. **AlphabetSoup 중앙은 Codex가 인수 보류 중이다**([student-acquisition-contract](../student-acquisition-contract/README.md)):
   teacher 품질은 close 중 물체 XY 이동을 접근 변위 제한에서 빼지만, student guard는 0.5초 연속
   접촉 전까지를 접근으로 본다. 같은 후보의 배치 3건(yaw −15°, X ±20 mm)도 같은 문제가 있을 수
   있으므로 같은 대조군 검사가 필요하다. teacher 쪽 기준은 바꾸지 않았다.
2. **품질 여유가 얇은 승인:** Peaches 중앙(힘 28.9 N·침투 1.96 mm)과 X +20 mm(힘 28.86 N·침투
   1.9988 mm), Raisins 중앙(침투 1.99970 mm)과 yaw +15°(침투 1.99467 mm). 기준은 통과했다.
3. 세 물체는 이번 탐색에 쓴 개발 물체다. 독립 validation/test가 아니다.

teacher 실행만 했고 student 학습·평가·visual/mask/goal-memory, lecun `dexprompt-x0-bc`는
건드리지 않았다.

## 남은 프로세스와 디스크

- 실행 중인 파이프라인 프로세스·tmux 세션 없음(05:59 종료 확인). 다른 사용자 프로세스는 건드리지 않았다.
- 디스크 여유 5.4 GB(공용 루트 파일시스템). 같은 물체의 동일 모델은 hard link로 공유돼 있다
  (내용·해시 불변). 격리 디렉터리 `out/hope-pilot/quarantine/`은 증거로 보존했다.

**Large Diameter는 정렬 뒤 재확인에서 전부 탈락했다.** 정렬 자체는 66/66 성공했고
IK도 65/66 통과했지만, 모든 자세가 테이블(world)과 겹쳐 `kept` 0이 됐다.
`align_grasps`는 남은 초기화가 0개일 때만 exit 1을 내므로(`return 0 if total else 1`)
이는 **빈 결과이지 실행 오류가 아니다**. v3 wrapper가 이 경우를 "alignment failed"라고
기록한 것은 이름이 부정확하다(차단 자체는 맞다 — 정련할 대상이 없다).

| seed | 자세별 가장 깊은 테이블 겹침 | 가장 흔한 겹침 링크 |
| --- | --- | --- |
| 0 | 최소 2.1 · 중앙 21.0 · 최대 32.5 mm | 엄지 끝 `finger_r_link4` 55/66 (중앙 21.3 mm), 손 base `hx5_r_base` 32/66 (중앙 8.9 mm) |
| 1 | 최소 5.6 · 중앙 20.2 · 최대 40.3 mm | 엄지 끝 56/66 (중앙 20.4 mm), 손 base 36/66 (중앙 9.4 mm) |

matcher 4회는 이미 테이블 여유를 지키려고 줄인 설정이며 계획에 고정돼 있어 바꾸지
않았다. 같은 Large Diameter가 AlphabetSoup에서는 seed당 44–48개를 남겼으므로,
Peaches의 형상에서 이 유형의 엄지 접근이 테이블 아래로 향하는 것으로 보인다(원인
분리는 하지 않았다). 따라서 Peaches의 동적 실행 후보는 Power Disk에서만 나온다.
