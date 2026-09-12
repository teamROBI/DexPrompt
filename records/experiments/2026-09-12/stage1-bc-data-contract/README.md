# 1단계 BC 데이터 계약 감사와 인계 (Codex용)

2026-09-12. 1단계 모델: 정확한 물체 형상·배치·로봇 상태를 입력으로 Dexonomy 성공 시연을 BC로 학습하고,
새 물체의 파지 시범을 생성하는 모델. (mask·point·language 입력 VLA는 2단계.)

- **읽기 전용 감사**다. 새 학습·렌더·물리 실행 없음. 모델 파일은 읽고 보관된 초기 상태만 `mj_setState`로 설정했다
  (시간 진행 없음). 공용 코드·기존 데이터·student 파일 수정 없음. 모든 항목 `student_ready=false`.
- 생성: `third_party/dexonomy/.venv/bin/python scripts/build_stage1_bc_handoff.py` (이 폴더를 다시 씀, 약 4분).
- 이 폴더는 `records/experiments/2026-09-12/stage1-bc-data-contract` (`results/`는 이 경로로의 링크).

## 1. 수량과 누락

`out/`와 `/data1/yspark/sim/icra`의 수집 폴더 401개를 전수 조사했다. accepted episode는 episode SHA 기준 **고유 128개**
(재현본·복사본 113개를 하나로 합침). **기록·자산 연결 누락 0**: 128개 모두 보관된 기록(메시·실제 scale·초기 상태·모델·
27관절 위치·실행 행동·Dexonomy 유형·후보·Dexonomy 원본 파일·manifest 행·scene)을 찾아 연결했고 기록 해시와 일치한다
(`issues.json` 빈 목록). 이는 **이미 보관된 값의 연결에 빠진 것이 없다는 뜻으로만 한정**한다. 매 tick 물체 자세·관절 속도·
팔 중력 보상 항은 처음부터 저장되지 않았으므로(3절), 이 자료가 모든 관측 설계에 바로 학습 가능한 것은 아니다.
그런 값을 요구하는 관측 설계는 재생(replay) 등 별도 생성이 필요하며 이번 인계에는 없다.

| 묶음 | episode | 물체 (후보 수) | 비고 |
| --- | ---: | --- | --- |
| **기본 풀** (A 물리, 단일 물체, 현재 controller, visual profile 없음) | **77** | Raisins 38 (17), AlphabetSoup 23 (4), Peaches 14 (2), core_can 2 (2) | 후보 25, Power Disk 39 · Large Diameter 38, 81,161 프레임 |
| B 물리 (multiccd) | 3 | AlphabetSoup·Peaches 중앙, Milk | A와 섞지 않음 |
| 다물체 (방해물) | 5 | clutter 2, 목표 변경 3 | 단일 물체 풀과 분리 |
| 이전 controller | 42 | 병(Inferior Pincer) 41, 캔 1 | 관측·행동 기록 방식은 같으나 접근·닫기 제어가 다른 시연자 |
| visual profile | 1 | 병 | 영상 크기·머리 자세가 다름 |

- **기본 풀은 4물체**(AlphabetSoup·Peaches·Raisins + core_can)다. **주요 평가 대상은 HOPE 3물체**(AlphabetSoup·Peaches·
  Raisins, 75개)이며, core_can 2개는 이전 scene·scale(0.08) 계열의 보조 형상이다(6절의 실제 분할 포함 상태 참조).
- 기본 풀 배치: 중앙 25, x ±10·±20 mm, y ±10·±20 mm, yaw ±7.5·±15° (한 요인 변화).
- 같은 조건 키(물체·후보·scene 배치·episode 회전·grasp root 오프셋·물리·방해물·controller)에 서로 다른 episode가 둘 이상인
  경우 **0**. core_can 1개는 scene 회전이 아니라 episode 회전 15°(`episode_yaw_deg`)라 `placement`는 centre로 표기된다.

## 2. 시간 정렬·좌표·행동 의미 (실제 수집 코드와 대조, 128개 전부)

수집 코드 `TeacherExecutor.tick()`(src/icra/execution.py): 영상과 `right_state`를 **먼저** 기록하고, 그다음 `action`을 명령해
10 physics substep(0.02 s)을 진행한 뒤 그 구간의 접촉·힘·높이·이동 신호를 **같은 프레임에** 붙인다.

| 검사 | 통과 |
| --- | ---: |
| 프레임 수 = 기록된 tick 수 | 128/128 |
| `right_state[0]` = 보관 초기 상태의 27관절 값(float32, 비트 단위) | 128/128 |
| `timestamp[0]` = 초기 상태 시각, 간격 = `control_dt`(0.02 s) | 128/128 |
| `action[0, 7:]` = 후보의 열린 손 명령(첫 transit tick) | 128/128 |
| 모든 행동이 actuator 범위 안 | 128/128 |
| phase 순서 transit→descend→approach→arrive→close→lift→hold | 128/128 |
| 기록된 world 손 목표 = 초기 물체 자세 @ Dexonomy root @ root 오프셋 (1e-6 이내) | 128/128 |
| Dexonomy 손가락 관절 순서 = 실행기 `finger_r_joint1..20` 순서 | 확인 |

- **행동 계약**: 기록된 `action[i]`는 27 actuator(`arm_r_joint1..7`, `finger_r_joint1..20`)에 실제로 들어간 **최종 actuator 명령**이며
  [t_i, t_i+0.02 s] 동안 유지된다. 수집 코드 `TeacherExecutor.action()`이 팔 관절 목표에 그 순간의 중력 보상 `qfrc_bias/kp`를 더하고,
  27개 전체를 actuator `ctrlrange`로 **clipping한 뒤**의 값이다(손가락 20개는 위치 목표 rad, 보상 없음). 보상 항은 따로 저장되지 않았다.
  - 이 값을 BC 정답으로 쓰면, 학습된 출력은 **그대로 actuator 명령(`data.ctrl`)으로 넣어야 한다. 실행 경로에서 중력 보상을 다시 더하면
    안 된다**(보상이 두 번 들어간다). clipping도 이미 반영돼 있다.
  - 반대로 "순수 관절 목표"를 정답으로 원하면, 보상 항이 저장되지 않았으므로 기록에서 분리할 수 없다(재생 필요, 이번 범위 밖).
- **좌표**: world(MuJoCo, m). 로봇 base는 원점·고정, lift·머리는 scene 값으로 고정. 물체 자세는 world 위치 + 쿼터니언 wxyz.

## 3. 입력의 가용 시점 (`fields.json`)

| 시점 | 필드 | 입력 사용 |
| --- | --- | --- |
| 에피소드 시작 전 | 메시·scale·초기 물체 자세(모든 물체)·초기 27관절·scene, Dexonomy 목표, 실행 목표(world 손 목표·열린/닫힘 손가락 목표·접근 축) | 가능 (목표는 목표 조건부 모델에서만) |
| tick i 행동 직전 | `right_state[i]`, `rgbm[i]`(alpha = 시뮬레이터 목표 mask, 2단계 전용 특권 정보), `right_cam_img[i]`, `timestamp[i]` | 가능 |
| tick i 행동 | `action[i]` | 정답(label) |
| teacher 내부 | `phase[i]` | 관측 아님 — 입력 금지 권장 |
| tick i 행동 **이후** 구간 | 접촉군·힘·목표 높이·xy 이동·침투 | tick i 행동의 입력 금지 |
| 에피소드 결과 | accepted·quality·status·재현 판정·미래 상태 | 입력 금지 |
| **저장 안 됨 (누락)** | 매 tick 물체 자세(초기 자세와 tick별 높이·xy 이동량만 있음), 관절 속도, 팔 중력 보상 항, 매 tick 손 root 자세(right_state의 순기구학으로 계산 가능하나 여기선 하지 않음) | 추정으로 채우지 않음 |

## 4. Dexonomy 파지 목표 (episode별 `dexonomy_target`, `executed_target`)

- **원본**: 후보 행의 `source` `.npy`(SHA 확인). 키: `grasp_qpos`(1×27), `squeeze_qpos`(1×27), `pregrasp_qpos`(128개 모두 6×27),
  `required_cbody`(필수 접촉군 5×2 링크), `hand_cbody`, `obj_cpn_w`/`hand_cpn_w`(10×6 접촉점 xyz + 법선), `obj_pose`, `obj_scale`, `tmpl_name`.
- **좌표·단위**: Dexonomy 물체 좌표계(물체 asset 자체 좌표, 최저점 z=0, `obj_scale` 적용). qpos[0:3] root 위치 m, [3:7] root 쿼터니언 wxyz,
  [7:27] 손가락 20관절 rad.
- **실제 실행**: world 손 목표 `world_root`; 손가락은 `open_qpos`로 접근해 `squeeze_qpos[7:]`를 목표로 접촉 피드백 닫기;
  pre-grasp는 **실행기 자체 경로**(목표에서 접근 축 반대로 0.10 m 물러난 지점, 그 위 0.15 m) — Dexonomy `pregrasp_qpos`는 기록만 있고 실행되지 않았다.
  실제로 통과한 접근 축·부호와 거부된 방향 사유도 기록.
- **두 모델의 구분**: (i) 목표 조건부 실행 모델 — 입력에 `executed_target`/`dexonomy_target` 사용 가능, 출력 `action`.
  (ii) 형상→목표 생성 모델 — 입력은 메시·scale·초기 자세(·로봇 상태), 정답은 `dexonomy_target`(grasp·squeeze·pregrasp·필수 접촉군);
  생성 목표의 실행 평가는 이후 별도 물리 실행이 필요하다(이번에 하지 않음).

## 5. 중복·관계·과거 사용

- 재현본은 같은 episode SHA라 하나로 합쳤고, 원장에 남은 생성 기록(원 실행·재현·대조)은 `produced_by`에 모두 남겼다
  (기본 풀: 재현 1회 52개, 2회 7개, 5–6회 3개, 없음 15개; 원장 기록 없는 2개는 core_can).
- `candidate_group` = 물체/후보: 같은 후보의 배치 변형을 묶는다(기본 풀 25개 그룹).
- **과거 평가 사용 표시**(`past_evaluation_use`): 기본 풀 77개 중 **60개**가 이미 다른 실험의 평가 조건이었다 —
  후속 배치 격자 49, 선정 비교 평가 풀 6, hope-pilot 배치 전이 5, candidate-diversity 배치 3, 평가 풀 보완 2(중복 표시 가능).
  선정기 학습 라벨로 쓰인 episode 9개(`past_training_use`). 선정 비교 평가 풀 후보에서 나온 episode 27개.

## 6. 분할 **제안** (`split-proposals.json`, 확정 아님)

단위는 episode 전체(같은 궤적의 프레임이 train/test에 섞이지 않음). 새 배치 평가와 새 형상 평가는 따로 구성했다.

| 제안 | train | test | 설명 |
| --- | ---: | ---: | --- |
| A. 학습 조건 재현 | 77 | — | 학습한 (물체, 후보, 배치) 조건에서 rollout. 적합 확인이지 전이 아님 |
| B1. 새 배치 — 안쪽 (±10 mm, ±7.5°) | 52 | 25 | test의 후보는 다른 배치로 train에 있음 |
| B2. 새 배치 — 바깥쪽 (±20 mm, ±15°) | 50 | 27 | 외삽 쪽 |
| C1. 새 형상 — AlphabetSoup 제외 | 54 | 23 | 물체의 모든 배치·후보를 제외 |
| C2. 새 형상 — Peaches 제외 | 63 | 14 | |
| C3. 새 형상 — Raisins 제외 | 39 | 38 | Raisins가 유일한 Large Diameter라 유형도 함께 빠짐 |

- 주의: B1·B2의 test는 **전부** 과거 후속 배치 격자에서 이미 평가된 조건이다(teacher 결과를 이미 봤다). 새 배치 test를 원하면
  아직 실행하지 않은 배치를 새로 정해야 한다. 형상은 3개뿐이라 C는 일반화 근거로 약하다.
- **정정 (core_can 포함 상태)**: 앞선 판의 "core_can 2개는 어느 제안에도 넣지 않았다"는 설명은 틀렸다. `split-proposals.json`의 실제
  내용은 core_can 2개가 **A·B1·B2·C1·C2·C3 모든 제안의 train에 포함**되고 어느 test에도 없다(위 표의 train 수에 포함).
  `split-proposals.json`의 `not_in_any_proposal` 키 이름은 이 실제 상태와 맞지 않지만, 분할 산출물은 바꾸지 않고 여기서만 정정한다.
  core_can을 train에서 뺄지는 분할 확정 때 정한다(빼면 각 train 수가 2씩 줄어든다).

## 7. 파일

| 파일 | 내용 |
| --- | --- |
| `episodes.jsonl` | 128개 전부: 경로·해시, 물체·메시·scale·초기 자세, 초기 관절, scene, 후보·유형, Dexonomy 목표, 실행 목표, 품질, 검사 결과, 풀/분리 사유, 생성 기록·과거 사용, 조건 키, 복사본 경로 |
| `fields.json` | 필드별 가용 시점·의미·누락 |
| `split-proposals.json` | 분할 제안(확정 아님), `candidate_groups` |
| `summary.json` · `issues.json` | 수량·검사 통과 수 / 누락·불일치(현재 0) |

## Codex 담당

1단계 모델 구현·학습, 시각 재렌더·mask 검사·student 인수. 이 폴더는 원본을 복사하지 않고 경로·해시로만 가리킨다.
