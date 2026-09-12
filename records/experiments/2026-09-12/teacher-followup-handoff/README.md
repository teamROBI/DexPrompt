# Teacher 후속 배치 시범 인계 (Codex용)

2026-09-12. 원본: `/data1/yspark/sim/icra/teacher-followup-5h-20260912-v1/HANDOFF.json`(49건).
이 디렉터리는 **참조 원장**이다. 원본 HANDOFF·원장·episode·student 데이터·코드는 삭제·수정하지 않았다.
생성: `third_party/dexonomy/.venv/bin/python scripts/build_followup_handoff.py` (읽기 전용, 물리 실행 없음).

## 핵심 수치

| 항목 | 값 |
| --- | ---: |
| 이번 HANDOFF 시범 (teacher accepted) | 49 |
| 그중 동일 조건 재현 통과 (episode SHA 동일) | 49 |
| 재현본을 독립 시범으로 센 수 | 0 |
| 추적 검사 전부 통과 | 49 / 49 |
| 누락·불일치 (`issues.json`) | 0 |
| 기존 HANDOFF와 **정확한 중복** (episode SHA 동일) | 5 |
| 같은 물체·후보·배치지만 기록이 다른 항목 | 0 |
| **순수 신규** | **44** |

정확한 중복 5건: AlphabetSoup `057c04e1d5` x_minus20mm·x_plus20mm·yaw_minus15, Peaches `203b95f140` x_plus20mm
(이상 `out/hope-pilot/HANDOFF.json`), Raisins `3e8fa79936` yaw_plus15(`hope-candidate-diversity-v1/HANDOFF.json`).
결정적 시뮬레이션이라 같은 매니페스트 행·scene이면 같은 episode가 나온다.

기존 자료 포함 고유 수 (재현 통과 teacher 시범만; 기존 17건은 모든 이전 HANDOFF의 재현 통과 항목):

| 단위 | 기존 | 이후 | 순증 |
| --- | ---: | ---: | ---: |
| episode | 17 | 61 | +44 |
| 후보 (물체, 후보) | 7 | 10 | +3 (`34815d4f21`, `7a8cdecf40`, `af15fdf0bc`) |
| 물체 | 4 | 4 | 0 (AlphabetSoup·Peaches·Raisins, 기존 Milk는 B 물리) |
| 물체-배치(조건·물리 포함) | 12 | 40 | +28 |
| 물체-후보-배치 | 17 | 61 | +44 |

- 기존 17건에는 clutter 2건(방해물 조건)과 Milk 1건(multiccd B 물리)이 조건/물리 키로 분리되어 포함된다. A 물리와 섞지 않는다.
- 새 후보 3개의 **중앙** 성공 episode(selector-compare-v1, 이번 단계1/중앙 재현)는 어떤 HANDOFF에도 없다. 이번 인계 대상은 배치 시범 49건뿐이다.

## 해석 조건 (반드시 함께 읽을 것)

- 6개 후보는 **학습 선정기의 선택이 아니다.** 기존에 중앙에서 accepted된 A 물리 단일 물체 후보 중, 원 episode의 품질 한계 대비 여유
  `min((침투한계−최대침투)/침투한계, (힘한계−최대힘)/힘한계)`가 큰 순(동률 해시 순, 근접 중복 제외)으로 물체당 2개를 고른 고정 규칙이다.
- 따라서 이 결과는 **중앙 성공 후보를 X/Y ±10·±20 mm, yaw ±7.5·±15°로 한 요인씩 옮긴 배치 평가**다. 같은 후보의 배치 다양성이지
  새 후보/새 형상 다양성이 아니며, 형상은 3개다.
- teacher accepted = 실행기 자체 기준(hold·필수 접촉군 5개·침투 2 mm·손가락 힘 30 N·테이블·들어올림 0.1 m) 통과. **student 인수 판정과 별개**이며
  모든 항목 `student_ready=false`, `student_acceptance="not checked (Codex)"`.

## 후보별 배치 결과 (72회 전부: `placements-72.csv`/`.json`)

| 물체 | 후보 | accepted/12 | accepted의 최소 여유 범위 | 실패 배치 (종료 단계) |
| --- | --- | ---: | --- | --- |
| AlphabetSoup | 057c04e1d5 | 9 | 0.091–0.286 | y−20 (lift), yaw+7.5 (hold), yaw+15 (close) |
| AlphabetSoup | 34815d4f21 | 8 | 0.003–0.068 | x−20 (hold), x−10 (lift), x+10 (hold), y−10 (lift) |
| Peaches | 7a8cdecf40 | 11 | 0.029–0.131 | yaw−15 (hold) |
| Peaches | 203b95f140 | 1 | 0.001 | x+20 외 11개 (hold 4, lift 2, close 5) |
| Raisins | af15fdf0bc | 11 | 0.258–0.344 | yaw−15 (hold) |
| Raisins | 3e8fa79936 | 9 | 0.134–0.267 | x−20, yaw−15, yaw−7.5 (모두 hold) |

- 실패 23건: hold 중 lift_timeout 12(접촉 획득 후 상실), lift 중 lift_timeout 5, close 중 no_quality_feasible_command 4·closure_timeout 2.
- accepted 49건은 모두 접촉 유지 ≥1.0 s. 단 **최소 여유 0.05 미만이 13건**(최저 0.001, Peaches `203b95f140` x+20·AlphabetSoup `34815d4f21`)으로
  한계 근처다. 표의 `margin_min_fraction`으로 걸러 쓸 수 있다.
- 열: 성공 여부, 실행기 종료 상태, 실제 마지막 단계(`end_phase`), 접촉 분류, 최장/종료 접촉 유지 시간, 최대 접촉군/필수 5,
  최대 침투·침투 여유, 최대 힘·힘 여유, 재현 판정, 중복 분류.

## 파일

| 파일 | 내용 |
| --- | --- |
| `manifest.json` | 49건. 항목마다 원 실행(UUID·collection·episode·model·초기 상태), 재현 실행(UUID·episode, 시범으로 세지 않음), scene·mesh·template·manifest 경로와 해시, 후보 행 해시, controller/collector 해시, compiled 물리 profile, 품질 여유, 중복 분류(`duplicate.class`, `counts_as_new_independent_episode`), 추적 검사 결과(`checks`) |
| `existing-reference.json` | 이전 HANDOFF 8개 파일의 해시와 참조한 기존 시범 17건(파일 존재·해시 재확인, 모델·초기 상태·물리·controller 기록) |
| `placements-72.csv` / `.json` | 6후보×12배치 전부 |
| `issues.json` | 누락·불일치 목록 (현재 빈 목록) |
| `summary.json` | 위 수치 |

추적 검사 내용(항목별 `checks`): episode·model·초기 상태·scene·manifest 파일 SHA 재계산 = 기록값, mesh·template 파일 SHA,
매니페스트 안 후보 행과 템플릿, 원 실행 원장 행(role placement, completed, accepted)과 attempt의 후보·모델·초기 상태·controller·collector 해시,
scene = 중앙 scene에 사전 고정 배치 한 요인만 적용한 것 + 사전 검사 준비, 후보가 고정 선정 목록에 있음,
재현 원장 행(role placement_reproduce, `reproduction_of` = 원 UUID, 같은 manifest·scene, accepted, episode SHA 동일, 파일 SHA 동일).

## Codex 담당 (이번에 하지 않음)

- 시각 재렌더, mask/가시성 검사, student 인수 검사.
- 인수 대상은 `manifest.json`의 `demonstrations[*].original.episode`(+ `model`, `initial_state`, `scene`). 중복 제외가 필요하면
  `duplicate.class == "new"`(44건)만 사용하고, 한계 근처를 빼려면 `quality_margin.margin`으로 거른다.
- 모델 파일은 실행 폴더 안에서 동일 해시끼리 hard link로 공유된다(경로·해시는 그대로).
