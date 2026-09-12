<!-- research-scope-2026-09-12 -->
> **날짜별 실험 계획 기록:** 현재 목표·우선순위는 [연구 정의](../../RESEARCH_PLAN.md)와
> [현재 작업서](../../plans/CURRENT_WORK.md)가 우선한다. 아래 실행 지시·P번호·미실행/진행 상태는 작성 당시 것이다.
> 고정 후보 접근 PPO는 형상 기반 선정/적응 RL 전체와 다르다. 부위 지시는 필수 범위가 아니다.
> 완료된 작업을 이 문서만 보고 재시작하지 않는다. 기존 승인 및 원본 보존은 유지한다.

# Claude 실행서 — 기존 HOPE 풀에서 새 파지 후보 검증

## 목적과 범위

이번에는 **새 teacher 후보를 확보할 수 있는지, 기존 후보보다 품질 여유가 있는지**
제한된 비용으로 확인한다. 새 형상 일반화 실험은 아니다. 이미 생성한 HOPE 정련·선별
풀을 이용하며 GraspGenX/align/refine, 새 물체, 그릇/망치 튜닝은 시작하지 않는다.

Student 최종 x0는 0/3이고 GPT-5.6이 진단·데이터 통합을 맡는다. Student/visual/mask/
goal-memory·학습 코드와 기존 데이터/checkpoint는 수정하지 않는다.

현재 teacher8건 중 student 인수7건이다. AlphabetSoup 중앙만 보류이며 같은 후보의
yaw−15°,X±20mm 세 시범은 이미 통과했다. 이 검사를 다시 요청하거나 실패라고 추정하지 않는다.

## C0. 기존 결과의 실행 비용을 정정하고 출력 위치를 준비한다

먼저 읽기:
- `results/2026-09-11/consolidated-status/README.md`.
- `results/2026-09-11/hope-centre/README.md`, `out/hope-pilot/HANDOFF.json`.
- `out/hope-pilot/run-config.json`, `generation-plan.json`의 실제 경로
  `out/hope-pilot-v1/generation-plan.json`.

1. **48회는 최종 유효 시도 수**다. 다음 격리 경로에 실제 episode ID 4개가 있다:
   `out/hope-pilot/quarantine/20260911T052609/out/hope-pilot/Raisins/executed/attempts.json`.
   최종48 + 인프라 오류에 포함된 실제 실행 최소4 + 재현8을 구분한다.
   네 건은 독립 실패로 성공률 분모에 다시 더하지 않지만 계산 비용·재시도에서 빼지 않는다.
   잘린 다음 시도처럼 기록이 불완전한 시작 횟수는 별도 unknown/partial로 남긴다.
2. 새 ledger는 경로만 다시 읽어 계수하지 않는다. attempt UUID, object/placement,
   candidate hash, 시작 시각, stage/collection, 실제 simulation 시작 여부,
   terminal outcome, output hash, infrastructure error, retry-of를 보존한다.
   실제 물리 시작은 재시도해도 사용한 예산이다. infra failure를 이유로 예산을 되돌리지 않는다.
   접근 실패도 실행 프로토콜의 시도 분모에 포함한다. 입력 검사 거절은 별도다.
3. 새 출력은 `/data1/yspark/sim/icra/hope-candidate-diversity-v1`처럼 data1에 만든다.
   repo `out/`에는 링크만 둘 수 있다. 기존 출력/HANDOFF/원본 모델은 덮어쓰지 않는다.
   텍스처 모델당 약210MB였으므로 최대출력·임시쓰기까지 공간을 계산한다.
   동일 바이트 모델 공유가 필요하면 hash 확인 후 immutable 파일끼리만 공유한다.
4. 실행 전 다른 작업·디스크·CPU/GPU를 확인한다. CPU worker2 이하, BLAS1, nice15.
   기존 실행환경에서 CPU teacher 실행을 유지해도 된다. GPU는 이번에 새 생성이 없으므로
   필요하지 않다. 작업은 tmux, 단계 exit/timeout/전체 로그와 종료한 자기 PID를 남긴다.

## C1. 원본 후보를 고정한다 — 실행 전에 manifest 작성

각 물체의 `out/hope-pilot/<object>/manifests/`에 있는 유형·seed별 선별 manifest를
`build_expansion_manifest.pooled()`와 같은 귀속 보존 규칙으로 읽는다.
기존 합친 상위12 manifest만 읽으면 새 풀이 사라지므로 원래 유형/seed별 풀을 사용한다.

- 물체: AlphabetSoup, Peaches, Raisins의 **중앙** 조건.
- 유형: 기존 계획의 두 유형만 유지. Soup/Peaches: Power Disk와 Large Diameter,
  Raisins: Palmar Pinch와 Large Diameter.
- 후보는 source 내용 SHA로 중복 제거한다. 실제 파일 hash가 manifest와 다르면 중단한다.
- 같은 물체에서 이미 실행한 후보는 성공/실패/인프라 중단 여부와 무관하게 제외한다.
  초기12 후보, 배치 재사용 후보, 격리 시도까지 포함한 exclusion manifest를 먼저 만든다.
- **물체당 새 후보 최대4개.** 두 유형 각각 hash순 앞2개를 기본 배분한다.
  한 유형이 부족하면 남은 유형의 아직 안 뽑힌 hash순 후보로 채운다.
  유형 순서는 위 나열 순서로 고정하고 round-robin으로 실행한다.
  같은 SHA가 여러 출처에 있으면 모든 parents를 보존하되 실제 실행은 한 번이다.
- 실행 결과를 보고 다른 후보로 교체하지 않는다. 풀이 부족하면 실제 개수와 이유를 쓴다.
- 기존 중앙 성공3개를 별도 control manifest에 둔다. 새 후보 통계에 넣지 않는다.

완료 산출물: `plan.json`, `candidate-pool.json`, `excluded.json`, `selected.json`.
후보 수/유형/seed·scene/mesh/template/source hash,정렬규칙,예산,마감시각을 고정한다.
이번 규칙은 이전 전체 hash상위12와 다르므로 기존1/12와 공정한 수율 개선 비교라고 하지 않는다.

## C2. 제한된 실행과 동일 재현

모든 실행은 home→접근→닫기→리프트→hold 전체 경로이며 기존 제어와 성공 조건 그대로다.
scene YAML에 yaw가 있으면 episode `--yaw 0`, 중복 회전 금지.
기존 contact-feedback closure, 마찰/질량/접촉강성/침투/힘/접촉군/시간 예산을 바꾸지 않는다.

| 단계 | 물체당 최대 | 합계 최대 | 다음 단계 진입 조건 |
| --- | ---: | ---: | --- |
| 기존 중앙 성공 control 재실행 | 1 | 3 | 원본 episode와 같은 성공·SHA |
| 새 중앙 후보 실행 | 4 | 12 | 해당 물체 control 정상 |
| 첫 새 중앙 성공 후보 재현 | 1 | 3 | 실행 순서상 첫 성공; 같은 상태·SHA |
| 새 후보 배치 검증 | 2 | 6 | 새 중앙 성공 재현 정상 |
| 배치 성공 재현 | 2 | 6 | 각 배치 성공 시 한 번 |
| **전체 실행 상한** | | **30** | 인프라 중단/재시도도 실제 시작하면 차감 |

- 기존 control이 실패/해시 불일치면 해당 물체는 신규 실행을 멈추고 원인을 기록한다.
  다른 물체의 독립 작업은 진행할 수 있다. 조용히 다른 control로 바꾸지 않는다.
- 새 중앙 성공이 여러 개면 전부 기록하지만, 확대 후보는 실행 순서상 **첫 성공 하나**다.
  결과를 보고 품질 여유가 가장 큰 후보를 골라 유리한 배치 실험을 만들지 않는다.
- 새 후보의 배치는 **yaw +15°, X −20mm** 두 조건으로 사전 고정한다.
  물체마다 유리한 방향으로 바꾸지 않는다. 기존 첫 후보도 이 조건에서 실행했으므로
  기존 기록과 나란히 볼 수 있으나 작은 표본의 후보 비교일 뿐 일반화 성공률은 아니다.
- 새 중앙 성공이 없으면 그 물체 확대는 미실행한다. 남은 예산을 다른 후보/물체/튜닝에 쓰지 않는다.
- accepted는 모든 기존 기준을 만족할 때만. 품질 여유가 작다고 실패로 다시 분류하거나
  반대로 기준을 완화하지 않는다. 여유와 accepted는 별도 지표다.
- 실행창은 시작부터 **최대6시간**, T+4h 이후 새 후보/배치 시작 금지.
  남은 시간은 이미 시작한 실행의 종료·재현·보고에 쓴다. 더 일찍 끝나면 종료한다.
  이것은 6시간을 채우라는 지시가 아니다.

## C3. 품질 여유와 인계를 남긴다

각 시도에 대해 다음을 같은 기준으로 기록한다:
- outcome·접근/닫기/운반/품질 위반 사유, 최대/종료 접촉군, 동시 접촉 지속 시간.
- 최소 높이 신호 기준 상승·접촉을 유지한 상승 지속 시간. hold라는 phase 이름만으로
  정지한 채 유지했다고 쓰지 않는다.
- 전체 tick 최대 침투·손가락 힘·테이블 충돌·획득 전 이동량과 각 한계까지 여유.
- 침투1.9mm 이상/힘29N 이상 tick 비율은 **서술용 근접 한계 지표**로 추가할 수 있다.
  새 accepted gate로 사용하지 않는다. 초기 몇tick만 보지 말고 close/lift/hold를 분리한다.
- 기존 후보와 새 후보의 접촉 요구/유형이 다를 수 있으므로 손가락 수가 다른 force합을
  직접 우열로 해석하지 않는다. 여러 여유를 임의 가중합해 "강건성 점수"라고 부르지 않는다.

보고할 질문:
1. 새로운 고유 teacher 후보의 완주가 생겼는가?
2. 기존 후보보다 힘/침투 여유가 나은 사례가 있는가? 서로 다른 여유의 trade-off는 무엇인가?
3. 같은 두 배치에서 어떤 차이가 있었는가? n이 작아 미결정인 부분은 무엇인가?
4. 다음에 새 형상을 늘릴 근거가 생겼는가, 아니면 접촉 유지가 여전히 병목인가?

새 `HANDOFF.json`은 **원본 HOPE HANDOFF와 분리**하고 accepted episode·model·initial
state·scene·mesh·template·candidate/parents·재현해시·원본control 대응을 포함한다.
Teacher accepted는 student ready가 아니다. Student 획득 계약 검사는 GPT-5.6에 인계한다.
GPT의 고정7건 묶음에 새 자료를 직접 추가하지 않는다.

최종 보고는 한국어로 `results/2026-09-11/hope-candidate-diversity/README.md`에 쓴다.
수정한 코드의 관련 회귀검사·Ruff와 기존control 확인을 보고한다. 작업에 영향 없는
전체 lint 오류를 고치느라 범위를 넓히지 않는다. 종료 시 자신의 tmux/PID 상태를 확인한다.
공유 `PAPER_TODO.md`, `RESUME.md`, 날짜별 index의 통합은 GPT-5.6에 맡긴다.
