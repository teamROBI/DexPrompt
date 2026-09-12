# Dexonomy reference 손목 편집의 물리 실행 가능성 — 당일 판정

2026-09-12, **완료 03:20** — zero-edit 6/6 원본 재현, 고정 손목 편집 24회 완주 0(실패 단계 이동은 있음), 물리 30/33. 작업서 [TODAY_FEASIBILITY](../../../plans/TODAY_FEASIBILITY_2026-09-12.md).
T0 02:43:06 KST, 새 실행 중단 07:43:06(T0+5h), 종료 09:43:06.
기존 A 물리, 기존 IK/경로 검증·접촉 피드백 close→lift→hold·품질 기준 그대로. 새 학습 없음.
이것은 **고정된 편집 제안 탐색**이며 RL 학습 결과가 아니다.

## 이전 작업의 중단 상태

[7시간 준비서](../../../plans/CLAUDE_SEVEN_HOURS_2026-09-12.md) 작업은 사용자 우선순위 변경(02:41)으로 멈췄다.
완료분과 재개 명령은 `/data1/yspark/sim/icra/selector-edit-prep-v1/status.json`에 있다.

- S0 시작 기록 완료.
- S1 AlphabetSoup guard 감사: **원인 미분리.** 계측 wrapper(`scripts/audit_guard.py`)의 `mj_geomDistance`
  인자 형식 버그로, 세 재실행 모두 시뮬레이션이 시작된 뒤(안정화·접근 후) 첫 닫기 확정 tick에서 멈췄다.
  guard 기록 없음, 원본 대조 불가. 물리 3회 차감(이 문서의 33회 예산과 별도). 버그는 수정했고 재실행하지 않았다.
- S2 선정 원장: 6개 원장 125행을 생성했다. 입력 행 수와 기록 행 수가 모두 일치했고, 표본 6행을 원본 episode
  해시까지 역추적했으며, 규칙 검사 7개가 통과했다. 마지막 필드 추가 뒤의 재생성은 하지 않았다.
- S3 선정기(`scripts/grasp_selector.py`): 코드만 작성했다. 실행·검사하지 않았고 학습하지 않았다.

## 편집 adapter

- [collect_wrist_edit.py](../../../../scripts/teacher/collect_wrist_edit.py): DemoFunGrasp `generate_reaching_plan_idx`와 같은 규칙으로
  편집한다. 물체 초기 좌표계에서 손목 reference를 **왼쪽 곱**으로 바꾸고(물체 좌표계 z축·원점 기준 회전 후 이동),
  실행기가 받는 `object_target`에 한 번만 적용한다.
  - 그 한 자세에서 접근 경유점(overhead·retreat·approach·arrive)과 닫기·리프트 목표가 모두 나오므로
    접근과 파지 reference가 어긋나지 않는다.
  - 리프트는 편집된 파지 자세 위의 상대 상승이라 원 변위가 유지된다. home은 그대로다.
  - 손 관절 목표·유형·필수 접촉군은 바꾸지 않는다. 편집값과 행렬을 `<collection>.wrist_edit.json`에 기록한다.
- 공통 실행기 코드는 수정하지 않았다(실행 범위 wrapper).
- 검사 `tests/test_wrist_edit.py` 5개 통과: 편집 0은 비트 단위 동일, 이동은 물체 좌표계 기준, yaw는 물체 축 기준
  위치·방향 회전, 알 수 없는 편집 필드 거부, 실행기가 편집된 reference를 받음.
- DemoFunGrasp one-step PPO(`ppo.py:run`, `storage.py:compute_returns`)와의 대응: 에피소드 시작에 편집 한 번 →
  전체 실행 → 결과. 이번에는 고정된 8개 편집만 실행하고 정책·보상 학습은 하지 않는다.

## 고정 조건 (실행 전, 02:47 `plan.json`)

규칙: 선정 원장에서 물체마다 A 물리·단일 물체·시뮬레이션 시작·인프라 오류 아님·라벨 있음 행 중 원장 순서로
첫 성공과 첫 실패. 모두 첫 HOPE 실행의 중앙 배치 후보다.

| 물체 | 성공 대조군 | 실패 조건 | 실패의 원본 결과 | 원본 시작 상태 보관 |
| --- | --- | --- | --- | --- |
| AlphabetSoup | `057c04e1d5` Power Disk | `00e490bec3` Power Disk | `no_validated_home_path` (시범 기록 전 접근 경로 검증 실패) | 아니오 — 초기 형식 |
| Peaches | `203b95f140` Power Disk | `0238ebfac6` Power Disk | `closure_timeout` | 예 |
| Raisins | `0ddf3447c0` Large Diameter | `015fe8b8ed` Large Diameter | `no_quality_feasible_command` | 예 |

AlphabetSoup 실패는 시작 상태와 episode가 보관돼 있지 않다. 그래서 zero-edit 재현은 종료 상태와 6개 접근 방향 판정
전체가 원본과 같은지로 판단한다. 나머지는 episode 해시와 판정으로 대조한다.

편집 8개(모든 실패 조건 공통, 순서 고정, 물체 초기 좌표계): x ±5 mm, y ±5 mm, z +5 / +10 mm, z축 회전 ±5°.

예산: zero-edit 6 + 편집 24 + 첫 개선 재현 최대 3 = **최대 33회**. 접근 실패도 분모와 비용에 포함한다.
성공과 실패의 zero-edit가 모두 원본을 재현한 물체에만 편집을 적용한다.

## 결과

### zero-edit (02:47:57–02:54, 6회, 인프라 오류 0)

**6/6이 원본을 재현했다.** adapter의 편집 0 경로는 기존 행동과 판정을 보존한다.

| 물체 | 조건 | 결과 | 원본과 비교 | 소요 |
| --- | --- | --- | --- | ---: |
| AlphabetSoup | 성공 `057c04e1` | completed, accepted | episode 해시·판정 동일 | 95.6 s |
| AlphabetSoup | 실패 `00e490be` | `no_validated_home_path` | 종료 상태·6개 접근 방향 판정 모두 동일 (원본 episode 없음) | 26.4 s |
| Peaches | 성공 `203b95f1` | completed, accepted | episode 해시·판정 동일 | 60.1 s |
| Peaches | 실패 `0238ebfa` | `closure_timeout` | episode 해시·판정 동일 | 97.8 s |
| Raisins | 성공 `0ddf3447` | completed, accepted | episode 해시·판정 동일 | 59.7 s |
| Raisins | 실패 `015fe8b8` | `no_quality_feasible_command` | episode 해시·판정 동일 | 41.3 s |

실행당 26–98 s(평균 약 63 s). 이 속도로 편집 24회는 약 25분, 첫 개선 재현을 포함한 완료는 03:25 전후로 예상했다.

### 편집 (02:54–03:19, 24회, 인프라 오류·timeout 0)

**편집 24회 중 완주(accepted) 0건.** 첫 개선이 없어 재현 3회는 쓰지 않았다. 판정은 실행기의 기존 기준 그대로이며,
실패를 다시 분류하거나 기준을 완화하지 않았다. 실제 종료 단계와 접촉 분류는 별도 열이다.

| 물체 (zero-edit 결과) | 편집 | 결과 | 실제 종료 단계 | 접촉 분류 | 동시 최대/종료 접촉군 | 최대 상승 | 접촉 유지 상승 | 최대 침투 / 힘 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| AlphabetSoup (접근 경로 실패) | x +5 mm | 접근 경로 실패 | 접근 | — | — | — | — | — |
| | x −5 mm | closure_timeout | close | 모두 동시 성립 없음 | 4/1 | 1.3 mm | 0 | 1.42 mm / 7.2 N |
| | y +5 mm | 접근 경로 실패 | 접근 | — | — | — | — | — |
| | y −5 mm | lift_timeout | lift | 성립 후 상실 | 5/3 | 37.2 mm | 0 | 1.89 mm / 27.8 N |
| | z +5 mm | 접근 경로 실패 | 접근 | — | — | — | — | — |
| | z +10 mm | closure_timeout | close | 모두 동시 성립 없음 | 4/1 | 3.0 mm | 0 | 1.99 mm / 29.8 N |
| | yaw +5° | closure_timeout | close | 모두 동시 성립 없음 | 4/3 | 2.6 mm | 0 | 1.79 mm / 16.4 N |
| | yaw −5° | 접근 경로 실패 | 접근 | — | — | — | — | — |
| Peaches (close 시간 초과) | x +5 mm | lift_timeout | lift | 성립 후 상실 | 5/2 | 7.6 mm | 0 | 1.95 mm / 28.8 N |
| | x −5 mm | no_quality_feasible_command | close | 모두 동시 성립 없음 | 3/3 | 0.4 mm | 0 | 1.96 mm / 23.9 N |
| | y +5 mm | no_quality_feasible_command | close | 모두 동시 성립 없음 | 3/3 | 0.3 mm | 0 | 1.99 mm / 25.0 N |
| | y −5 mm | closure_timeout | close | 모두 동시 성립 없음 | 4/4 | 2.6 mm | 0 | 2.00 mm / 29.0 N |
| | z +5 mm | lift_timeout | lift | 성립 후 상실 | 5/4 | **137.8 mm** | 1.34 s | 1.96 mm / 25.5 N |
| | z +10 mm | lift_timeout | **hold** | 성립 후 상실 | 5/4 | **149.0 mm** | 1.78 s | 1.95 mm / **30.0 N** |
| | yaw +5° | 접근 경로 실패 | 접근 | — | — | — | — | — |
| | yaw −5° | no_quality_feasible_command | close | 모두 동시 성립 없음 | 3/3 | 0.2 mm | 0 | 1.97 mm / 27.2 N |
| Raisins (`no_quality_feasible_command`) | x +5 mm | no_quality_feasible_command | close | 모두 동시 성립 없음 | 2/2 | 0.6 mm | 0 | 1.90 mm / 15.7 N |
| | x −5 mm | closure_timeout | close | 성립 후 상실 | 5/3 | 0.3 mm | 0 | 2.00 mm / 23.2 N |
| | y +5 mm | closure_timeout | close | 성립 후 상실 | 5/2 | 1.5 mm | 0 | 2.00 mm / 24.2 N |
| | y −5 mm | no_quality_feasible_command | close | 성립 후 상실 | 5/4 | 1.1 mm | 0 | 1.99 mm / 16.8 N |
| | z +5 mm | no_quality_feasible_command | close | 모두 동시 성립 없음 | 4/3 | 0.6 mm | 0 | 1.99 mm / 14.4 N |
| | z +10 mm | no_quality_feasible_command | close | 모두 동시 성립 없음 | 3/3 | 0.6 mm | 0 | 1.56 mm / 13.1 N |
| | yaw +5° | no_quality_feasible_command | close | 모두 동시 성립 없음 | 3/3 | 0.6 mm | 0 | 1.81 mm / 11.4 N |
| | yaw −5° | closure_timeout | close | 모두 동시 성립 없음 | 4/3 | 4.6 mm | 0 | 2.00 mm / 15.4 N |

("접근 경로 실패"는 시범 기록 전 접근 경로 검증 실패이며, 시뮬레이터를 시작해 예산에서 차감됐다. AlphabetSoup 원본과
편집 실패의 방향별 사유는 대부분 `ik_failure`, 나머지는 `quality_penetration`이었다. Peaches yaw +5°는 6방향 모두 `ik_failure`였다.)

관찰 — 편집은 실패 단계를 바꿨지만 완주로 바꾸지 못했다:

- **AlphabetSoup**: zero-edit는 접근 경로에서 실패했다. 편집 8개 중 4개(x −5, y −5, z +10, yaw +5)는 접근을 통과해
  close까지 갔다. y −5 mm는 lift에 들어가 37 mm까지 올랐다. 나머지 4개는 원본처럼 접근 경로에서 실패했다.
- **Peaches**: zero-edit는 close 시간 초과였다. z +5 mm·z +10 mm는 137.8·149.0 mm까지 들었고, z +10 mm는 hold 단계에
  들어가 접촉 유지 상승이 1.78 s였다. 그래도 종료 시 접촉군 하나를 잃어(5→4) 기준의 접촉 유지 hold를 채우지 못했다.
  z +10 mm의 최대 손가락 힘은 30.0 N으로 한계에 닿아 있다.
- **Raisins**: 8개 모두 close에서 끝났다. 원본(`no_quality_feasible_command`)과 같은 단계다.
- 편집별 종료 단계 합계: 접근 5, close 15, lift 3, hold 1.

## 오늘의 판정

작업서의 규칙표에서 **"zero-edit 정상, 편집 개선 0"**에 해당한다.

- 이번에 시험한 작은 손목 이동·회전(x/y ±5 mm, z +5/+10 mm, yaw ±5°)으로는 세 실패 조건 모두 완주로 바뀌지 않았다.
  **이 좁은 행동 공간은 이 세 조건에서 완주 개선을 보이지 않았다.**
- 편집이 결과에 반응하지 않은 것은 아니다. AlphabetSoup의 접근 통과 4/8, Peaches의 리프트 진입 3/8처럼 실패 단계가
  바뀌었다. 이는 편집 공간이 결과에 영향을 준다는 서술적 관찰이며, 성공이나 수율 개선이 아니다. 조건당 편집 1회씩이다.
- 이 결과로 DemoFunGrasp 전체(손 모양 편집 포함)나 시범 편집 RL 연구 자체가 불가능하다고 쓰지 않는다.
- 규칙표의 다음 일: 손 관절 목표(손 모양) 편집까지 포함한 행동 공간, 또는 생성 경로 변경. 같은 후보의 손목 미세 편집을
  더 늘려 반복하지 않는다(남은 예산 3회를 추가 편집에 쓰지 않았다).

## 비용·시간·상태

| 항목 | 값 |
| --- | --- |
| 물리 실행 | **30 / 33** (zero-edit 6, 편집 24, 재현 0). 인프라 오류·timeout 0 |
| 실행 시간 | 02:47:57–03:19:45 (총 1,907 s, 평균 63.6 s/회) |
| 이전 계획의 물리 실행 | 3회 (S1 guard 감사, 별도) |
| 완료 시각 | 03:20 — T0+5h(07:43)보다 일찍 끝났다. 규칙상 남은 시간·예산을 새 물체·유형·편집·튜닝으로 채우지 않는다 |
| 남은 프로세스·tmux | 없음(확인) |

산출물 `/data1/yspark/sim/icra/hope-wrist-edit-v1`: `plan.json`(조건·편집·예산 사전 고정), `ledger.json`, `runs.json`
(실행별 종료 단계·접촉 분류·품질 여유), `summary.json`, `HANDOFF.json`(accepted 0, `student_ready=false`), `runs/`(각 수집과
`*.wrist_edit.json`), `logs/`, `stages.json`, `chain-status.tsv`.

다시 만들기: `third_party/dexonomy/.venv/bin/python scripts/run_wrist_edit.py report --out /data1/yspark/sim/icra/hope-wrist-edit-v1`.
같은 조건의 새 실행은 새 출력 폴더에 `plan` → `run`으로 한다(고정된 plan은 다시 쓰지 않는다).
