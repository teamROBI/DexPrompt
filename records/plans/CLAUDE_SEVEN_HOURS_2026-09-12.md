# Claude 7시간 작업 — 선정 학습과 시범 편집의 실행 기반

> 최신 사용자 요청으로 [당일 판정 작업서](TODAY_FEASIBILITY_2026-09-12.md)가 우선합니다. 아래는 이전 준비 계획이며 그대로 재개하지 않습니다.

사용자 요청: 지금부터 약7시간 동안 중간 승인 대기로 멈추지 않고 다음 연구 단계를 준비한다.
실제 시작 시각 T0를 기록하고 T0+7시간을 종료 시각으로 삼는다. 절대 시각을 추정하지 않는다.
이 문서는 [기존 짧은 선정 준비서](CLAUDE_SELECTOR_PREP_2026-09-12.md)의 다음 작업을 확장한다.
연구 정의는 [RESEARCH_PLAN](../../RESEARCH_PLAN.md)을 따른다.

**이번 목표: 진단 보고에서 끝내지 않고, 실제 기록으로 학습 가능한 선정 데이터·모델/트레이너 연결·
zero-edit 실행까지 검증된 시범 편집 환경을 인계한다. 새 학습은 실행하지 않는다.**
7시간을 학습 승인으로 해석하지 않는다. 학습은 향후 사용자 지시에 따라 lecun에서 실행한다.
가상 완성 논문의 수치/보상/모델 세부사항은 실측도 합의된 구현 사양도 아니다.

## 시간 배분과 산출물

| T0 이후 | 작업 | 완료물 |
| --- | --- | --- |
| 0:00–0:15 | 인계·현재 작업/자원·원본 해시 확인 | 이번 실행 manifest, deadline, 변경/보존 목록 |
| 0:15–1:00 | AlphabetSoup guard 감사 | 최대3회 계측, 원인/미분리 판정, 기존 가설 정정 |
| 1:00–2:30 | 선정 학습용 원장·입력 계약 구축 | 원시 출처를 추적하는 데이터와 누락/중복/단계별 집계 |
| 2:30–4:00 | 선정 모델·비교·트레이너 연결 | 학습 없이 실제 배치 forward/loss·순위 출력·분할 검사 |
| 4:00–6:00 | 최소 시범 편집 환경 구현 | edit=0 동일성, bounded edit 계약, full-execution reward/종료 연결 |
| 6:00–7:00 | 통합 검사·누락 보완·인계 | 관련 테스트, 실행 명령, 결과/미완료/다음 학습 gate |

시간은 배분 기준이다. 앞 단계가 빨리 끝나면 다음 단계로 즉시 넘어간다.
1시간 이후 guard 미분석을 이유로 남은6시간을 같은 후보 튜닝에 쓰지 않는다.
마지막1시간에는 새 장시간 물리 실행을 시작하지 않고 검사와 인계에 집중한다.

## 공통 작업 경계

- Codex 담당: 기존 lecun 시각 BC 결과 확인, Raisins clutter2건의 student 인수.
  `src/icra/student*`, visual/mask/goal-memory, 기존 student config/data/cache/checkpoint는 수정하지 않는다.
- 기존 A12와 모든 teacher 원본/재현 파일은 보존한다. Milk B는 계속 분리한다.
- 공통 실행기/접촉 제어/물리·quality 기준은 바꾸지 않는다. 새 adapter/wrapper/독립 모듈로 구현한다.
  기존 API로 불가능한 부분은 추가 학습이나 광범위 리팩터링 대신 구체적인 차단 사유로 남긴다.
- `PAPER_TODO.md`, `RESUME.md`, `CURRENT_WORK.md`, 원고는 Codex가 관리한다. 결과와 새 작업서만 작성한다.
- 별도 새 출력: `/data1/yspark/sim/icra/selector-edit-prep-v1` (이미 존재하면 새 suffix).
  모델은 원본 경로와 해시를 참조하고 불필요하게205MB MJB를 복제하지 않는다.
- CPU worker 최대2, BLAS1thread, nice15를 시작점으로 한다. 모델 검사는 가능한 CPU에서 한다.
  다른 사용자의 프로세스나 Codex tmux를 중지하거나 기존 GPU를 임의 점유하지 않는다.
- 긴 명령은 별도 tmux. 실행기/기록기는 stage exit·timeout·duration·실제 물리 시작 여부를 기록한다.
  timeout은 자신의 child process group만 종료하며 부분 산출물은 격리/표시한다.
- 새 학습, 새 GraspGenX 예측, 정련 확대, 물체/배치 탐색, 물리 기준 완화는 이번 범위에 없다.

## S0. 시작 기록

현재 디렉터리, git 상태, 실행 중 프로세스/tmux, 사용 가능한 디스크를 읽는다.
진행 중인 사용자 작업은 건드리지 않는다. 아래 파일과 원본 해시를 읽고 manifest에 남긴다.

- `RESEARCH_PLAN.md`
- `results/2026-09-12/hope-clutter-crosscheck/{README.md,audit.json}`
- `/data1/yspark/sim/icra/hope-clutter-v1/{HANDOFF.json,runs.json}`
- 기존 HOPE centre/candidate-diversity HANDOFF 및 실제 collection/episode_entries.

`status.json`에는 started_at, deadline, active_stage, completed_stages, blocked_stages,
training_started=false, physical_runs_started, 산출물 경로를 남긴다. 완료와 실행 중을 혼동하지 않는다.

## S1. guard 원인 확인 — 최대45분, 물리 최대3회

Codex가 보관 source/NPZ를 확인했다. AlphabetSoup 실패2건의 최대 외력은0.002712/0.383212 N,
close 외력 override 임계값은22.5 N, 요구 접촉군은 목표만 센다.
“방해물 힘을 목표 접촉으로 오인”했다는 가설은 지지되지 않는다.
두 실패의 `guard_scaled_ticks=560`은 더 직접적인 단서지만 어떤 거부인지 아직 모른다.

- 단일 대조1 + 실패2만 보관 source/model/initial state로 계측할 수 있다.
- 기록 command만 재생해 내부 preview 결정을 측정한 것으로 쓰지 않는다.
- 원본과 적용 action/상태/기존 물리 신호가 같아야 진단을 해석한다.
- 최초 거부와 축소 반복 구간의 arm/finger fraction, 예측 접촉 body pair·침투,
  실제 명령 변화·목표까지 거리·contact/force를 기록한다.
- body 원점이나 convex 조각 중심을 손가락 접촉 pad 거리로 대체하지 않는다.
- 원인을 확인하면 종료한다. 원인 미분리/재현 실패라도 시간·실행 상한에서 멈추고 다음으로 간다.
- controller 수정이나 손가락을 더 조이는 비교는 실행하지 않는다.

## S2. 선정 학습 데이터

새 스크립트는 입력 root/manifest를 인자로 받고 JSONL/정적 메타데이터를 새 디렉터리에 쓴다.
한 행은 후보와 장면의 실제 실행이며, 다음을 보존한다.

1. 출처 경로/해시, 원래 attempt ID, object/mesh/scale/배치, 로봇 초기 상태, 방해물,
   type/candidate/접촉군, 실행기/물리/config 식별자.
2. 전체 physical outcome, approach/close/carry의 reached와 success, phase별 미관측 마스크.
3. 실제 수행 비용, 재현/재시도/중복, 인프라 오류와 부분 실행 표시.
4. 입력 시점에 사용할 수 있는 후보 자세/유형/정적 특징과 결과에서만 얻는 label을 분리.

필수 규칙:

- 미실행 후보는 실패가 아니다. 접근 실패 후 닫기·리프트 라벨은 미관측이다.
- 인프라 오류는 물리 실패 라벨이 아니다. 결과를 잃은 실행 비용은 비용 원장에서 유지한다.
- 동일 후보의 재현은 새 독립 표본으로 부풀리지 않는다. 원본을 삭제하지 않고 관계를 남긴다.
- 같은 mesh의 scale/texture/배치는 동일 geometry group. scene만 다른 경우 형상 일반화로 쓰지 않는다.
- A/B와 제어기 버전을 명시적으로 구분한다. 다른 버전의 성공/실패를 하나의 동일 과제로 몰래 합치지 않는다.
- 현재 성공3형상만으로 일반화 성능을 주장하지 않는다. 각 fold의 class/label 가용성을 확인한다.

먼저 HOPE A를 완성한다. 조기에 끝나면 병/캔의 과거 기록도 **동일 계약 여부를 감사**해 별도 cohort로
인덱싱한다. 호환성을 증명하지 못하면 학습 pool에 넣지 않는다. 새 물리 수집은 하지 않는다.

완료 조건: 원장 수치가 입력 collection 합계와 맞고, 일부 행을 원본까지 역추적하며,
누락 출처/실행 안 된 단계/재현을 구분하는 회귀 검사가 통과한다.

## S3. 선정 모델과 무학습 연결 검사

최소 모델을 구현하되 기존 기록의 정보량에 맞춘다. 대규모 backbone이나 과도한 새 의존성은 피한다.

- 입력: 정확한 형상 특징 또는 결정론적 표면 점 샘플, 배치/로봇 상태, 후보 자세/유형/접촉 구조,
  방해물 표현. 학습에 사용 가능한 입력과 누락 입력을 명시한다.
- 출력: full-execution score와 선택적으로 reached-stage 조건부 score.
  refinement pass나 생성확률을 full-execution label로 사용하지 않는다.
- loss: stage 관측 마스크를 적용하고, 미래 접촉력/최종 grasp pose/결과 파일명으로 답을 누설하지 않는다.
- baseline: 고정 seed 무작위 순위와 결과를 보지 않는 정적 순위. 특징이 없으면 없는 것으로 보고한다.
- train/dev 분할과 normalization은 geometry group 기준으로 만들고 split을 체크포인트 계약에 저장한다.
- `--check`는 실제 원장 batch를 읽고 forward/loss·유한성·마스크·candidate ranking을 확인하되
  backward/optimizer.step이 호출되면 실패한다. 입력/모델 해시도 남긴다.
- train entrypoint와 resume/checkpoint/evaluate 경로는 연결하되 이번엔 train을 호출하지 않는다.

**평가 편향 주의:** 원장에서 일부 후보만 실행했으면 새 랭커의 top-k가 미실행 후보일 수 있다.
이들을 실패 또는 성공으로 채워 실제 수집 수율을 추정하지 않는다. 완전 관측 후보 subset에서의
진단 순위, 관측 coverage, 미관측을 명시한 평가만 허용한다. 무학습 순위/fixture 정확도를
학습 성능으로 보고하지 않는다. 다음 실제 비교는 후보 pool·query budget을 고정해야 한다.

완료 조건: 실제 데이터에서 input→score/loss→ranking→저장/읽기 경로 확인,
optimizer 갱신0, 중복/누수/미관측/NaN 처리 검사 통과, 실행 가능한 학습 명령 준비.

## S4. 최소 시범 편집 환경 — 추가 물리 최대3회

Dexonomy의 유형/후보 선정을 DemoFunGrasp식 편집과 구분한다. 이 단계는 아직 학습하지 않는다.

- 먼저 편집할 reference의 정의와 작은 action 집합을 고정한다. 기존 executor가 명확히 지원하는
  pregrasp/approach offset부터 시작할 수 있다. 가상 논문의32차원을 구현 의무로 보지 않는다.
- 좌표계·단위·action bounds·적용 시점과 원래 teacher 접촉 구조의 유지/변경 규칙을 문서화한다.
- one-step 환경: initial privileged observation + candidate → edit 한 번 → full execution → reward/outcome.
  한 tick 시뮬레이션과 혼동하지 않는다. contact-feedback close/lift/hold는 그대로 사용한다.
- reward는 완료·실패/품질·수정 크기·비용을 분리해 반환한다. 수치 계수는 가설로 기록하며
  acceptance는 기존 독립 판정에서 가져온다. 단순 위치 접근을 full success로 바꾸지 않는다.
- edit=0에서 기존 명령/상태/품질/종료 판정을 유지해야 한다. 실제로 학습을 돌려 확인하지 않는다.

실제 실행은 기존 HOPE 단일 물체 기준선3개에 대해 **zero edit만 각1회**, 합계3회 이내다.
S1에서 얻은 동일 실행 결과를 유효하게 재사용할 수 있으면 재실행을 줄인다.
합성 fixture로 비영점 편집의 좌표변환·bounds·reward/terminal contract를 검사한다.
이번 블록에서는 물리적으로 비영점 edit를 탐색하지 않는다.

환경 연결에 필요한 독립 모듈과 체크 스크립트를 만든다. 접촉 제어를 바꾸지 않고 구현할 수 없으면
partial adapter와 필요한 변경 목록을 남긴다. 이 때문에 S2/S3 인계를 포기하지 않는다.

완료 조건: zero-edit baseline 동일성 또는 구체적 차단 사유, 입력/행동/보상 명세,
미관측/실패/timeout 처리, 향후 PPO 연결에 필요한 인터페이스와 한정된 다음 검증 명령.

## S5. 통합 검사와 인계

1. 새/변경 모듈과 직접 의존 경로의 관련 tests, Ruff, diff 공백 검사를 수행한다.
   기존 전체 suite가 실패하면 관련성/원인을 구분하고 무관한 광범위 수정을 하지 않는다.
2. 원본/모델/데이터 보존을 확인한다. 기록된 출처로 각 산출물을 재생성할 수 있어야 한다.
3. `results/2026-09-12/selector-edit-prep/README.md`에 다음을 구분한다.
   - guard 원인 확인/미분리, 물리 실행 실제 횟수와 결과.
   - 원장 전체 수·독립 후보/형상 수·라벨/누락 분포.
   - 선정기 연결 통과/실패, 학습 미실행, 편집 zero-baseline 검증 범위.
   - 바로 실행할 다음 명령, 추가 데이터/승인/구현이 필요한 조건.
4. 출력 폴더에 HANDOFF.json과 최종 status.json을 쓴다. `training_started=false`를 유지한다.
   독립 검증 없이 student_ready/generalization_ready를 true로 쓰지 않는다.
5. 자신의 프로세스가 종료됐는지 확인한다. 남길 필요가 있으면 PID/tmux/출력/중지 방법을 명시한다.

총 물리 상한은 **6회 = S1 최대3 + S4 최대3**다. 시작한 실행은 오류/재시도도 차감한다.
어느 물체의 대조가 불일치하면 그 분기의 물리 해석을 멈추되 데이터/모델 구현은 계속한다.

## 빨리 끝나거나 막혔을 때의 우선 대체 작업

7시간을 같은 실험 반복으로 채우지 않는다. 다음 순서로 아직 남은 구체적 작업을 수행한다.

1. 입력 경로를 바꿔도 생성 결과가 같은지, 원장 캐시/해시/중복 복원이 되는지 검사.
2. 후보 라벨 미관측·fold에 양성 없음·빈 후보 pool·중복 부모 등 실제 위험 사례 보강.
3. 실제 사용 가능한 shape/type 입력과 결과 누설 위험의 데이터 사전 작성.
4. 각 단계의 손실 입력, 모델 크기/추론 시간, feature 추출 비용을 무학습으로 계측.
5. 다음 제한된 학습 실험을 시작하기 위한 gate와 명령을 한 문서로 고정.

범위 내 오류는 고치고 계속 진행하며 중간 승인을 기다리지 않는다.
기준 변경이나 새 대규모 실험이 필요한 지점은 구체적으로 남기고 다른 독립 작업을 진행한다.
모든 완료 조건과 대체 작업까지 끝났다면 억지 실행/대기를 하지 말고 실제 완료 시각과 이유를 보고한다.
