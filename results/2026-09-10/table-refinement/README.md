# 테이블을 고려한 정련: 소규모 A/B

날짜: 2026-09-10 · 학습은 시작하지 않았습니다.

정련(ADMM)에 테이블을 넣으면 무엇이 달라지는지, 같은 입력·같은 템플릿·같은
예산으로 한 쌍씩 비교했습니다. 결론부터: **정적 검사의 테이블 관통 기각은
4건 → 0건으로 사라졌고 실행에 도달한 후보는 3개 → 7개로 늘었지만, 완주는
1건 → 1건으로 그대로입니다.** 늘어난 4개는 모두 이후 단계에서 실패했고,
실패 지점이 "접촉을 아예 못 잡음"에서 "다 잡았다가 놓침"으로 옮겨갔습니다.
기존 캔 성공은 바이트 단위로 보존됐습니다.

## 1. 테이블 제약이 빠지는 위치

DGN의 장면 설정(`scene_cfg/<obj>/floating/scale0XX.npy`)에는 물체 하나뿐이고
평면이 없습니다. Dexonomy 자체는 평면을 완전히 지원합니다.

| 위치 | 평면을 쓰는가 |
| --- | --- |
| `dexonomy/data/obj_loader.py:107-113` | `plane` 항목 → `col_plane` |
| `dexonomy/op/gen_init.py:102` | `op=init`이 `check_plane`으로 걸러냄 |
| `dexonomy/op/gen_grasp_admm.py` | `col_plane` 참조가 **하나도 없음** |
| `dexonomy/sim/mujoco_env.py:52-58, 238-257` | 평면을 margin 있는 geom으로 생성 |
| `dexonomy/sim/mujoco_env.py:388-396` | 손–테이블 접촉은 world body(id 0)라 ho/hh 어디에도 안 들어감 |

즉 ADMM에는 평면 항목이 직접 들어가는 자리가 없고, 테이블은 오직 최적화용
MuJoCo 환경의 물리 접촉을 통해서만 작용합니다. 손을 밀어내되 물체 접촉으로는
세지 않는 동작이라 원하던 그대로입니다.

그런데 장면 설정에 평면을 넣으면 정련이 0건이 됐습니다. 원인은 물리가 아니라
두 가지 기계적 문제였습니다.

1. **상대 경로.** `load_scene_cfg`(`dexonomy/util/file_util.py`)는 `*_path`를
   **그 설정 파일이 있는 디렉터리**에 이어 붙입니다. 다른 곳에 쓴 사본은
   없는 메시를 가리키고, `safe_wrapper`가 `ParseXML` 오류를 삼켜서 결과는
   "빈 출력 디렉터리"로만 보입니다.
2. **단일 항목 단정.** `gen_grasp_admm.py:25`의
   `assert len(scene_cfg) == 1, "Only single object is supported now."` —
   테이블을 넣는 순간 항목이 둘이 되어 후보마다 여기서 죽습니다.

기억해 두었던 "장면 설정에 테이블 평면을 넣으면 수확이 0이 된다"는 관찰은
테이블에 대한 증거가 아니라 이 두 문제의 흔적이었습니다. 이번 비교의 첫
실행(`out/tabletop/run-pathbug.log`, `out/tabletop/manifests-pathbug/`)도 같은
이유로 0건이었고, 그 수치는 폐기했습니다.

### 같은 물리적 테이블인가

물체 프레임에 놓은 평면과 실행기가 세우는 테이블이 같은 면인지 확인했습니다
(`out/tabletop/table-check.json`).

| 물체 | 정련이 보는 원점 높이 | 실행기 | 차이 | 법선 각도 |
| --- | --- | --- | --- | --- |
| bottle | 0.102284176325 m | 같음 | 4.2e-17 m | 0 rad |
| camera | 0.022307041104 m | 같음 | 5.9e-17 m | 5.2e-08 rad |
| can | 0.037887066876 m | 같음 | 3.5e-17 m | 0 rad |

## 2. 고정한 조건

- 물체·자세: can `core_can_7bd0557…` scale 0.08, camera `core_camera_1ab3abb…`
  scale 0.08, 둘 다 **rest 0 · yaw 0**. 놓임 자세·yaw·닫기 제어·성공 기준은
  이번에 함께 바꾸지 않았습니다.
- 템플릿: can `10_Power_Disk`, camera `22_Parallel_Extension`. GraspGenX는
  다시 돌리지 않았습니다.
- 입력: 물체마다 **12개**를 내용 sha256 순서로 고정하고 실행 **전에**
  `out/tabletop/inputs.json`에 기록했습니다. can의 12개에는 과거 성공으로
  이어진 입력(`0_96_grasp.npy`)이 포함돼 있습니다(`led_to_earlier_success`).
- 두 팔이 정말 같은 파일을 봤는지는 보고 스크립트가 확인하며, can·camera 모두
  `same_initialisations: true`입니다.
- 선택·실행 설정: `select --limit 64`, `collect --max-candidates 64
  --max-yaws 1 --seed 0 --width 64 --height 64`.
- 네 팔 모두 같은 코드 상태에서 돌렸습니다. 패치는 평면이 없는 장면에서
  아무 일도 하지 않습니다(테스트로 고정).

카메라 12개 중 2개는 처음 복사할 때 다른 seed의 같은 파일명을 덮어써서 초기
실행이 10개만 봤습니다. seed를 붙인 이름으로 복원해 두 팔 모두 12개로 맞춘
뒤 실행했고, 경위는 `out/tabletop/inputs.json`의 `note`에 남겼습니다.

## 3. 단계별 결과

`results/2026-09-10/table-refinement/funnel.json`

### 캔

| 단계 | 테이블 없음 | 테이블 있음 |
| --- | --- | --- |
| 초기화 입력 | 12 | 12 |
| 정련 성공 (정련 오류) | 7 (0) | 7 (0) |
| 정적 검사 `hand_through_table` | 4 | **0** |
| 정적 검사 `kept` | 3 | **7** |
| 실행 시도 / 고유 후보 | 3 / 3 | 7 / 7 |
| 접촉군을 한 번도 다 못 잡음 | 2 | 1 |
| 다 잡았다가 놓침 | 0 | 5 |
| **완주** | **1** | **1** |

시도별로(접촉군 5개 기준, `max` = 동시 최대, `all-held` = 전 접촉군 동시 유지
tick, `end` = 종료 시점 접촉군):

| 입력 | 테이블 없음 | 테이블 있음 |
| --- | --- | --- |
| `0_26` | (정적 기각: hand_through_table) | closure_timeout · max 5/5 · all-held 52 · end 4 · 8.6 mm |
| `0_57` | (정적 기각) | no_quality_feasible_command · 5/5 · 108 · 2 · 3.9 mm |
| `0_69` | no_quality_feasible_command · **3/5** · 0 · 3 · 2.4 mm | lift_timeout · **5/5** · 382 · 2 · **147.3 mm** |
| `0_75` | (정적 기각) | closure_timeout · 5/5 · 58 · 4 · 2.2 mm |
| `0_86` | (정적 기각) | no_quality_feasible_command · 3/5 · 0 · 2 · 2.3 mm |
| `0_96` | **completed** · 5/5 · 493 · 5 · 108.7 mm | **completed** · 5/5 · **596** · 5 · 108.7 mm |
| `0_9` | no_quality_feasible_command · 4/5 · 0 · 2 · 3.2 mm | lift_timeout · 5/5 · 61 · 4 · 4.9 mm |

기각·중단 사유의 뜻: `no_quality_feasible_command`는 품질 게이트를 통과하는
명령을 못 찾아 접근/닫기가 중단된 경우(`src/icra/execution.py:447`),
`closure_timeout`은 닫기 구간이 끝날 때까지 리프트 준비가 안 된 경우
(`execution.py:577`), `lift_timeout`은 리프트에 들어갔지만 유지 조건을
채우지 못한 경우(`execution.py:579`)입니다. 반복 시도는 없었고(시도 수 =
고유 후보 수), 접근 실패도 분모에 그대로 들어 있습니다.

정련 자세 자체가 움직인 것도 확인했습니다. 같은 입력에 대해 손목이
11.7–51.2 mm, 관절이 최대 0.24 rad 이동했습니다. 테이블이 화면 뒤 검사에만
쓰인 게 아니라 최적화에 실제로 작용했습니다.

### 카메라

| 단계 | 테이블 없음 | 테이블 있음 |
| --- | --- | --- |
| 초기화 입력 | 12 | 12 |
| 정련 성공 (정련 오류) | 1 (0) | **0** (0) |
| 정적 검사 | `open_hand_inside_object` 1 | — |
| 선택 / 실행 / 완주 | 0 / 0 / 0 | 0 / 0 / 0 |

두 팔 모두 정련 오류가 0이므로 이번 0건은 설정 문제가 아니라 최적화가 실제로
유효한 파지를 못 찾은 결과입니다. 카메라는 이 템플릿·이 크기에서 테이블
유무와 무관하게 거의 아무것도 통과하지 못합니다.

## 4. 테이블 충돌 감소가 완주를 늘렸는가 — 따로 본 판단

**아니오.** 정적 검사의 테이블 관통 기각은 4 → 0으로 완전히 사라졌고 실행에
도달한 후보는 3 → 7로 늘었지만, 완주는 1 → 1입니다. 늘어난 4개(`0_26`,
`0_57`, `0_75`, `0_86`)는 모두 이후 단계에서 실패했습니다.

바뀐 것은 **실패 지점**입니다.

- 테이블 없음: 완주하지 못한 2건은 접촉군을 한 번도 다 잡지 못했습니다
  (최대 3/5, 4/5). 리프트도 2–3 mm에 그쳤습니다.
- 테이블 있음: 완주하지 못한 6건 중 5건이 5/5를 잡았고, 그중 `0_69`는
  147.3 mm까지 들어 올린 뒤 2/5로 떨어졌습니다. 실패가 "획득 실패"에서
  "유지 실패"로 옮겨갔습니다.

정련에 테이블을 넣는 것은 후보의 질을 실제로 높였지만(접촉을 잡는 데까지는
간다), 완주를 막는 다음 병목은 닫기·유지 쪽입니다. 이번에는 지시대로 닫기
제어와 성공 기준을 함께 건드리지 않았으므로, 그 병목은 다음 실험거리로
남깁니다. 표본이 물체 하나·입력 12개이므로 완주 1 대 1의 차이 없음 자체를
결론으로 일반화하지는 않습니다.

## 5. 기존 캔 성공 보존

- 기준선 팔이 과거 성공을 **바이트 단위로 재현**했습니다: episode sha256
  `b4ddabc29bd969e8e582c61e5d9a095abe1b7f6611d16fba3b813645cdb91a9e`,
  후보 `9d7009a7406f7304…`, 입력 `0_96_grasp.npy`. 이번 코드 변경 뒤에도
  같은 값입니다.
- 같은 입력이 테이블 팔에서도 완주했습니다(다른 정련 자세이므로 episode
  sha는 `1628a6c7cd2d66da…`로 다름). 전 접촉군 동시 유지가 493 → 596 tick으로
  늘었습니다.
- 과거 결과 파일(`out/multiobject/`, `results/2026-09-09/`)은 건드리지
  않았습니다.

동일 조건 재현도 확인했습니다. 테이블 팔을 같은 manifest·seed로 다시 돌린
결과(`out/tabletop/tab_can_table_reproduce/`) 7개 시도가 상태와 episode sha256
모두 완전히 일치했습니다.

| 시도 | 상태 | sha256 |
| --- | --- | --- |
| `teacher_0000` | closure_timeout | `c0b473af8e3c59d0…` |
| `teacher_0001` | no_quality_feasible_command | `e9d26e4016066bb2…` |
| `teacher_0002` | lift_timeout | `37fbef6a26856fe2…` |
| `teacher_0003` | closure_timeout | `060c903f9d17e7d3…` |
| `teacher_0004` | no_quality_feasible_command | `811259f0e692ac48…` |
| `teacher_0005` | **completed** | `1628a6c7cd2d66da…` |
| `teacher_0006` | lift_timeout | `b2398cf9651906fa…` |

완주한 `0_96`은 teacher가 요구한 접촉군 5개를 596 tick 동안 동시에 유지했고
종료 시점에도 5/5였습니다(접촉 준수).

## 6. 바꾼 코드와 테스트

| 파일 | 내용 |
| --- | --- |
| `scripts/build_tabletop_scene_cfg.py` | `resolved_entries()` — 사본이 옮겨가도 찾을 수 있도록 `*_path`를 원본 설정 기준 절대 경로로 씀 |
| `scripts/dexonomy_table_scene/sitecustomize.py` | 지연 import 훅. `ICRA_DEXONOMY_TABLE_SCENE=1`일 때만 동작하며, 메시 로더가 고르는 항목만 바꿈 |
| `scripts/report_table_refinement.py` | 두 팔을 같은 규칙으로 세는 보고서 |
| `tests/test_table_scene.py` | 경로가 옮겨가도 유효한지, 상대 경로면 실패하는지, 평면이 있을 때 물체만 넘어가는지, 평면이 없으면 원본이 그대로 도는지, 물체 두 개는 여전히 거부되는지, 환경 변수 없이는 훅이 잠자는지 |
| `tests/test_table_refinement_report.py` | 획득 실패와 유지 실패 구분, 반복 시도·접근 실패가 분모에 남는지, 입력이 다르면 짝짓기가 거짓으로 표시되는지 |

`third_party/`(= `~/projects/Dexonomy` 심볼릭 링크)는 한 줄도 고치지
않았습니다. 패치는 실행 범위 안에서 함수 하나를 교체할 뿐이고, 정련기가
워커를 spawn으로 띄우기 때문에 `PYTHONPATH`에 얹은 site 훅으로 부모와 워커
모두에 적용됩니다. 처음에는 훅에서 곧바로 Dexonomy를 import했는데, 인터프리터
시작 시점에 MuJoCo가 딸려 들어오면서 GLFW 탐색이 표준 입력을 읽어 워커가
멈췄습니다. 그래서 import 시점까지 미루는 방식으로 바꿨습니다.

## 산출물

| 경로 | 내용 |
| --- | --- |
| `out/tabletop/inputs.json` | 실행 전에 고정·기록한 입력 목록 |
| `out/tabletop/table-check.json` | 정련용 평면과 실행기 테이블의 일치 확인 |
| `out/tabletop/run.log` | 네 팔의 실행 기록 |
| `out/tabletop/run-pathbug.log`, `out/tabletop/manifests-pathbug/` | 폐기한 첫 실행(경로 버그) |
| `out/tabletop/manifests/*.verdicts.json` | 정적 검사의 후보별 판정 |
| `out/tabletop/tab_can_base/`, `tab_can_table/` | 시도별 npz와 `attempts.json` |
| `out/tabletop/tab_can_table_reproduce/` | 같은 조건 재실행(7/7 sha 동일) |
| `results/2026-09-10/table-refinement/funnel.json` | 단계별 집계 |
