# stage1-goal-bc-v1 자료 형식 (BC·PPO·DP3 공용)

위치: turing `/data1/yspark/sim/icra/stage1-goal-bc-v1/`, lecun `/data1/yspark/projects/dexprompt/stage1-goal-bc-v1/`.
원본 teacher 자료는 수정하지 않으며, 이 폴더는 경로·해시로 원본을 가리키는 사본/파생 자료다. 모든 항목 `student_ready=false`.

## package/ (turing에서 생성, lecun으로 `rsync -aL` 전송, 양쪽 `sha256sum -c MANIFEST.sha256` 일치 확인)

| 파일 | 내용 |
| --- | --- |
| `MANIFEST.sha256` | package 안 모든 파일의 sha256 (`sha256sum -c`로 검증) |
| `episodes.jsonl` | episode 75개, 한 줄에 하나 (아래 필드) |
| `models/<sha256>.mjb` | 보관된 컴파일 MuJoCo 모델 (이름 = 내용 sha256, 37개). 메시·텍스처 포함, 자산 없이 로드 가능 |
| `initial/<sha256>.npy` | 보관 초기 상태, `mjSTATE_INTEGRATION` float64 (이름 = 내용 sha256). `mj_setState`로 복원 |
| `arrays/<name>.npz` | 원본 episode 배열에서 영상만 뺀 사본 (아래 키) |

`episodes.jsonl` 필드: `name`, `episode_sha256`(원본 npz), `object_id`, `candidate_id`, `grasp_type`, `placement`, `candidate_group`,
`model`/`model_sha256`, `initial_state`/`initial_state_sha256`, `arrays`/`arrays_sha256`, `frames`, `required_contact_groups`(5×2 링크 이름),
`distractors`, `initial_object_poses`(물체별 world 위치 m + 쿼터니언 wxyz, 초기 상태 기준), `initial_joint_qpos`(27, rad), `scale`, `mesh_sha256`,
`dexonomy_target`(grasp/squeeze 1×27, pregrasp k×27 — Dexonomy 물체 좌표계, 위치 m·쿼터니언 wxyz·손가락 rad; required contact groups;
template; source sha256), `executed_target`(`world_root` 4×4 world m — 초기 물체 자세 @ Dexonomy root @ root offset; 열린 손 명령 20;
닫힘 목표 20; 실제 통과한 접근 축·부호·standoff·overhead — Dexonomy pregrasp는 실행되지 않음), `quality_limits`, `past_evaluation_use`.

`arrays/<name>.npz` 키 (T = 프레임 수, tick i = [t_i, t_i + 0.02 s]):

| 키 | 형태·dtype | 시점·의미 |
| --- | --- | --- |
| `right_state` | T×27 float32 | 행동 i **직전** 관절 위치(`arm_r_joint1..7`, `finger_r_joint1..20`), rad |
| `action` | T×27 float64 | tick i에 넣은 **최종 actuator 명령**(팔 = 관절 목표 + 중력 보상 `qfrc_bias/kp`, 전체 ctrlrange clip 후). 다시 보상을 더하지 말 것 |
| `timestamp`, `control_dt` | T float64 | t_i (s), 0.02 |
| `phase` | T str | teacher controller 내부 단계 이름 — 관측 아님(입력 금지) |
| `contact_groups` 등 신호 10개 | T×… | 행동 i **이후** 10 substep 구간의 집계(접촉군 all, 힘 min/max, 목표 높이 min, xy 이동, 침투 max) — 행동 i의 입력 금지 |

## runs/replay-*/ (lecun에서 재생해 생성, 원본과 분리)

재생: 보관 초기 상태 복원 → tick마다 `action[i]`를 그대로 넣고 10 substep 동안 `mj_step` → `icra.simulation.forward` → `ContactMeter.read`
(수집기와 동일 순서, 코드 사본 `code-v1`, sha는 `code-v1/CODE.sha256`).

| 파일 | 내용 |
| --- | --- |
| `replay-report.json` | episode별: `right_state_max_abs`·`first_state_mismatch_tick`(수치 차이), 신호별 `signal_max_abs`·`first_signal_mismatch`, `judge`(phase 없는 `contact_acquisition_guard_v2` 판정), `augment_sha256`, `source_arrays_sha256` |
| `augment/<name>.npz` | 행동 i **직전**에 읽은 미저장 상태 (T행, 원본에 없던 값) |

`augment/<name>.npz` 키: `time`(T, s) · `q`(T×27 float64, rad) · `qd`(T×27, rad/s) · `object_pos`(T×3, m, world) ·
`object_quat`(T×4, wxyz) · `object_vel`(T×6, free joint qvel: 선속도 m/s world + 각속도 rad/s 물체 좌표) · `hand_pos`(T×3, `scene.HAND_ROOT` body, m) ·
`hand_mat`(T×9, 회전행렬 row-major) · `arm_bias`(T×7, `qfrc_bias/kp`, 그 순간 팔 명령에 들어간 보상 항).
`q`의 float32 값은 `right_state`와 비트 단위로 같아야 한다(재생 검사 항목).

## 두 판정은 따로 기록한다

- **원본 teacher 판정 재현**: 수집기의 phase 기반 `icra.dataset.assess_episode` (teacher phase 사용). 원본 accepted가 그대로 나오는지.
- **phase 없는 계약** `contact_acquisition_guard_v2`: 학습 정책 평가용. 접촉 획득(0.5 s) 전 구간 전체에서 물체 xy 이동 ≤ 10 mm를 검사하므로,
  teacher의 '닫기' 구간 미끄러짐을 검사하지 않는 원본 판정과 결과가 다를 수 있다. 성공 기준은 어느 쪽도 완화하지 않는다.
- 수치 차이(최대 절대 차이)와 판정 일치 여부를 따로 적는다.
