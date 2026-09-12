<!-- research-scope-2026-09-12 -->
> **날짜별 실험 계획 기록:** 현재 목표·우선순위는 [연구 정의](../../RESEARCH_PLAN.md)와
> [현재 작업서](../../plans/CURRENT_WORK.md)가 우선한다. 아래 실행 지시·P번호·미실행/진행 상태는 작성 당시 것이다.
> 고정 후보 접근 PPO는 형상 기반 선정/적응 RL 전체와 다르다. 부위 지시는 필수 범위가 아니다.
> 완료된 작업을 이 문서만 보고 재시작하지 않는다. 기존 승인 및 원본 보존은 유지한다.

# GPT-5.6 실행서 — 시각 student 진단과 데이터 준비

## 임무와 완료 경계

Teacher 시범 생성은 가능해졌지만, 단일 병 x0 시각 정책은 최종0/3이다.
**현재 정책의 실패가 어디서 시작되는지 근거로 분리하고, 승인된 HOPE 7건을
lecun의 검증된 진단 데이터로 준비하라. 그 결과에 맞는 다음 단일 요인 실험을 설계하라.**

G0→G1→G2→G3를 수행한다. G2의 파일 확인·전송 준비는 G1 GPU 진단과 독립적으로
진행해도 된다. 불필요한 확인 질문 없이 이 범위를 완료한다. 이번 지시는 새 학습 실행을
포함하지 않는다. 새 학습은 G3의 구체적인 설정과 근거를 인계하는 지점까지 준비한다.
일반적인 코드 수정·무학습 검사·데이터 전송은 범위 안이다.

## G0. 상태를 읽고 현재 기준을 보존한다 (약 20–40분)

### 우선 읽을 파일

- `plans/HANDOFF_NEXT_2026-09-11.md`, `PAPER_TODO.md`.
- `results/2026-09-11/consolidated-status/README.md` 및 같은 폴더
  `x0-results.json`, `x0-diagnosis.json`, `x0-command-audit.json`, `peaches-lecun-check.json`.
- `results/2026-09-11/student-acquisition-contract/README.md`.
- `results/2026-09-11/student-action-precision/README.md`.
- `out/hope-student-pool-final.json`, `out/hope-pilot/HANDOFF.json`.

### 환경과 위치

| 항목 | 위치 |
| --- | --- |
| 로컬 프로젝트 | `/home/yspark/projects/ICRA` |
| 로컬 student Python | `.venv-student/bin/python` |
| 로컬 teacher Python | `third_party/dexonomy/.venv/bin/python` |
| 로컬 Ruff | `.venv/bin/ruff` |
| lecun | `yspark@147.47.200.171`, port `19423` |
| lecun 프로젝트 BASE | `/home/yspark/projects/DexPrompt` |
| x0 실행 코드 CODE | `BASE/out/student-x0-code-v1` |
| 원본 단일 병 데이터 | `BASE/out/student-single-bottle-v1` |
| 단일 병 cache | `BASE/out/student-single-bottle-v1-cache` |
| x0 모델 | `/data1/yspark/projects/dexprompt/student-runs/single-bottle-x0-v1/epoch_0012.pt` |
| epsilon 기준 모델 | `/data1/yspark/projects/dexprompt/student-runs/single-bottle-v1/epoch_0012.pt` |
| x0 평가 | `BASE/out/single-bottle-x0-job-v1/evaluation-epoch-13` |
| 복구된 물리 감사 | `BASE/out/x0-command-audit-recovery-v1/audit` |
| Peaches 데이터 / 검사 | `/data1/yspark/projects/dexprompt/student-{data,checks}/hope-pilot-v1/Peaches` |

표의 BASE/CODE는 설명용 약칭이다. 셸에서 그대로 실행하지 말고 절대 경로로 풀어 쓴다.
SSH 키는 로컬 `out/server-transfer/lecun/id_ed25519`에 있다. 파일 내용은 출력하거나
문서/저장소에 복사하지 않는다. 연결 예:

```bash
ssh -i /home/yspark/projects/ICRA/out/server-transfer/lecun/id_ed25519 \
  -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 \
  -p 19423 yspark@147.47.200.171
```

- 실행 전 git status, 관련 소스 해시, 기존 status/exit/tmux/PID, 디스크·GPU를 확인한다.
  다른 작업자의 dirty/untracked 파일은 정상적인 동시 작업일 수 있다. reset/clean 금지.
- 모델·전처리 해시 불일치를 `allow_code_change`로 우회하지 않는다. 원격 원본 코드를
  확인하고 진단 helper만 추가한다. 모델 변경이 필요하면 새 사본·새 실험으로 분리한다.
- 기존 x0/followup/HOPE watcher를 다시 시작하지 않는다. 명령 재생 감사도 이미 복구 완료다.
- 대용량 결과는 로컬 `/data1/yspark/sim/icra/`, lecun `/data1/yspark/projects/dexprompt/`.
  루트·`/tmp`가 같은 파일시스템이고 최근 실제 ENOSPC가 났다. 대상 디스크 여유와 예상
  최대 출력 크기를 비교한다. 타인의 파일·프로세스는 삭제/종료하지 않는다.
- 오래 걸리는 작업은 tmux, CPU는 BLAS/OMP/MKL1·nice15부터. GPU는 현재 점유를 보고
  1개 작업만 선택한다. 이전 GPU0 UUID는 `GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035`이며
  숫자 인덱스가 같은 장치인지 다시 확인한다. 새 학습이 나중에 승인되면 lecun에서만 실행한다.

완료: 새 `results/2026-09-11/student-next-diagnosis/plan.json`에 입력/소스/모델 해시,
검사할 질문·관측·seed·계산 상한을 **실행 전에** 기록한다. 경로가 이미 있으면 v2 등 새 경로.

## G1. 첫 폐루프 이탈을 분리한다 (약 1–3시간, 추가 학습 없음)

### 이미 끝난 검사를 반복하지 않는다

- 단일 병 974 tick,27관절,20ms,절대 actuator 명령,chunk64/execute8.
- x0/epsilon 각각6,331 updates의 최종 평가0/3. x0의 5 checkpoints 모두0/3.
- x0 최종은 품질 위반 없이 시간을 채워도 필요한 2군을 동시에 만들지 못함.
- 최종 저장 명령3회 물리 재생 신호차이0. 정규화·소스·관절 순서·명령 큐·표본 전처리 확인.
- 명령 정규화 float32 왕복만으로 기존 병/캔 성공이 깨지지 않았음.
- x0 학습 관측 frame0 첫 명령 최대오차0.158–0.170rad, frame450 약0.003rad.
  이것만으로 원인이 공변량 이동/비전/학습량/보상이라고 단정하지 않는다.

### G1-A. 저장된 자료로 먼저 분석

1. 최종 3개 `replay_*.npz`와 복구 감사의 `pre_action_state`를 읽는다. 원본 시범의
   action/right_state/phase와 비교하되 phase는 그래프·분석에만 사용한다.
2. 팔7/손20을 분리해 첫64tick과 전체974tick의 명령오차, 측정상태오차,
   명령 변화량,추종오차,접촉군,상승을 그림/JSON으로 남긴다. 최대값의 tick·관절 이름도 쓴다.
3. 첫8tick과 재계획 경계8/16/...를 표시한다. 큰 명령 변화가 chunk 내부인지 경계인지
   확인한다. 교사도 같은 방식으로 측정한다. **지금 rate limiter를 붙이지 않는다.**
4. 단순한 시간 지연과 경로 이탈을 구분하기 위해 동일 tick 비교와 함께,
   teacher 상태열에서 현재 상태에 가까운 진행 위치를 보조 진단으로 계산한다.
   관절 거리의 정규화·팔/손 가중치·검색 범위를 기록한다. 최근접 teacher index는
   분석용이며 정책/성공 판정에 주입하지 않는다. 최근접만으로 경로 유효성도 선언하지 않는다.
5. 초기 상태부터 오차가 큰지, 초기에 맞다가 재계획에서 누적되는지, 정상 접근처럼 보이나
   접촉만 놓치는지, 거의 다른 상태에 머무는지에 따라 관측을 분류한다.

### G1-B. 필요한 경우에만 고정 관측에서 무학습 비교

기존4frame 진단으로 설명이 부족한 경우 수행한다. 기본 후보 tick은
`0,8,16,32,64,200,450,700`, 최종 x0, seed1000–1002. 기존frame결과를 재사용할 수 있다.
총 **8tick×2관측경로×3seed ≤48 action chunk forward**를 상한으로 한다.

- 경로1: 원래 teacher 기록 관측에서의 생성.
- 경로2: 저장된 student 명령으로 정확히 복원한 closed-loop 상태의 관측에서의 생성.
- 두 경로 모두 목표 기억을 tick0부터 순서대로 갱신한다. 미래 wrist reference를
  미리 넣거나 한 frame만 불러서 현재 goal memory인 것처럼 사용하지 않는다.
- 통제된 paired seed 비교와 실제 과거 실행의 RNG 상태를 구분한다. 실제 명령과
  비트 동일을 주장하려면 generator state와 이전 호출/목표 참조 획득에 따른 queue clear까지
  보존해야 한다. seed를 각 tick마다 다시 넣은 비교는 별도의 통제 진단이다.
- 관측을 바꿨을 때 행동이 바뀌는 것은 민감도일 뿐 원인 증명이나 시각 grounding 성공이 아니다.
- 네트워크는 eval/no_grad, backward와 optimizer 호출은 오류 처리한다.
- 새 코드가 필요한 경우 `diagnose_single_student.py`, `audit_student_failure.py`,
  `StudentActor`의 기존 전처리/goal memory를 재사용한다. 또 다른 전처리 구현을 만들지 않는다.

완료: 실패가 시작되는 tick·관절·재계획 여부와 두세 가설의 지지/반대 근거,
불확실성을 `README.md`, `diagnosis.json`, 그림에 남긴다. 재현 가능한 오류가 나오면
최소 회귀 테스트로 고치되 이전 모델 결과는 보존하고 수정 전/후 범위를 명시한다.
학습 실패만으로 코드를 고치거나 새로운 방법의 효과를 주장하지 않는다.

중단: 모델/데이터 해시불일치, 기록 명령 재현실패, 관측 정렬불일치면 해석보다 그 오류를
먼저 해결한다. GPU가 부족하면 CPU 분석·G2를 진행한다. 자동으로 여러 모델/하이퍼파라미터를
탐색하지 않는다. G1 전체 새 진단은 3시간 상한, 넘으면 부분 결과와 정확한 남은 질문을 쓴다.

## G2. HOPE 7시범을 통합·전송한다 (약 1–3시간)

### 고정 입력

`out/hope-student-pool-final.json`의 `ready` 목록이 유일한 입력 목록이다.
원본 HANDOFF8개를 전부 읽어 성공이면 넣는 방식은 금지한다. AlphabetSoup 중앙은 제외.
예상:7시범/7466프레임/3물체/7물체-배치/3teacher/2유형/초기입력7개.

1. 현재 source/episode/reproduction/scene/mesh 해시를 다시 확인한다.
   `report_hope_student_pool.py`를 새 출력에 실행한다. readiness가 달라지면 원인을
   기록하고 조용히 다른 후보로 대체하지 않는다.
2. `prepare_student_diagnostic.py`로 목록의 visual 디렉터리7개를 새 묶음에 넣는다.
   기존 visual profile을 다시 적용하지 않는다. 원본은 그대로 두고 데이터 import 귀속 보존.
   예시(인자와 경로를 실제 manifest에서 조립):

   ```bash
   .venv-student/bin/python scripts/prepare_student_diagnostic.py \
     --input <승인된_visual_경로1> --input <나머지_6개_각각> \
     --out /data1/yspark/sim/icra/hope-student-bundle-v1
   ```

   이는 복사해 바로 실행하는 명령이 아닌 인자 형식 예시다. subprocess argv 리스트로
   ready 목록에서 생성하고 실제 실행 argv를 기록한다. 작업 재개 시 기존 출력을 덮어쓰지 않는다.
3. bundle의7건/7466프레임,원본 대비action/state/quality/condition 보존,중복 없음,
   모델·초기 상태 상대 경로가 유효한지 검사한다. 모두train인 diagnostic임을 명시한다.
   frame무작위split·같은후보배치의형상test 주장 금지.
4. `train_hx5_student.py prepare-cache`, `audit_visual_intake.py`,
   `evaluate_student_suite.py --check-controls`, `preflight --allow-train-only-diagnostic`.
   기존 개별 검사 증거는 재사용하되 bundle귀속/합치기 오류를 확인한다.
   7개 양성 명령 완주+7개 정지 비성공,정확한실행수를 기록한다.
5. bundle을 lecun의 새 `/data1/.../student-data/hope-bundle-v1`로 전송한다.
   파일목록·크기·SHA를 전후 비교하고 `.partial`에 복사→검증→최종이름으로 전환한다.
   큰cache는 원격에서 만들면 전송량을 줄일 수 있다. 코드·스크립트 의존성도 먼저 점검한다.
   이전 `audit_student_failure.py` 누락을 반복하지 않는다.
6. 원격에서 cache/preflight와 물리 대조군을 확인한다. 기존 x0/epsilon 코드는 그대로 유지한다.
   `check_hope_student_lecun.py`는 Peaches 고정 작업용 전제가 있으므로 무작정 재사용하지 말고
   새 bundle인자·샘플선택·후속대기조건을 검토한다. 새 helper 또는 명시적인 옵션으로 분리한다.
7. 실제DINO full-model forward는 **물체별 첫시범 home와 close/lift 표본**, 최대3batch,
   batch2로 실시한다. 모든7건을 DINO검사했다고 과장하지 않는다.
   출력[2,64,27],유한loss/action,모델불변,backward/optimizer0,fixture아님 확인.

완료: `bundle-release.json`에 원본→통합episode 대응,해시,분할,개수,실제원격경로,
각검사결과,미검사항목과 새학습없음을 기록한다. 실행 가능한 다음 명령은 파일로 남긴다.
새로운 시범을 Claude가 보내도 이번 고정7건 묶음에 자동으로 섞지 않고 다음 버전 후보로 둔다.

## G3. 다음 실험 한 개만 설계한다 (약 30–60분)

G1결과를 근거로 학습/추론/데이터 중 **한 요인만** 바꾸는 실험서를 작성한다.
현시점에 rate limit·잔차행동·비전 변경·학습량 증가 중 하나를 미리 정답으로 고정하지 않는다.
확정 버그가 있으면 버그 수정 검증을 방법 개선 실험과 구분한다.

필수 항목:
- 검증할 질문과 반증 가능한 관측; 왜 기존 실험이 답하지 못했는가.
- 기존 x0 최종 기준선,같은 단일병과 평가계약,초기상태·seed·모델선택규칙.
- 변경하는 정확한 한 요인,고정조건,학습/추론 예산,학습이 필요하면 lecun 실행명령.
- 진단 학습이면 첫 기술 실행은 고정1seed. 이것으로 학습seed강건성을 주장하지 않음.
- 양성/음성대조군,최종epoch선정,완주/접촉/품질기준,실패시종료와기록조건.
- 예상소요시간은 실측steps/s로 산출; 일정이 아니라 상한과 남는 작업으로 표현.

이번 작업은 여기서 끝낸다. **G3 새 학습 명령은 실행하지 않는다.** 이미 한 범위는
매번 승인 질문 없이 마치고, 사용자에게 결과·다음 실험서를 한 번에 인계한다.

## 파일 소유권과 최종 인계

- 수정 가능: student 진단/통합/전송 helper와 관련 tests,자기 결과 문서,
  `PAPER_TODO.md`, `RESUME.md`. 모델 수정은 오류의 재현·새버전경계가 있을 때만.
- Claude 영역: teacher 생성/필터/정련/실행wrapper와출력. 임의수정/프로세스종료금지.
- 기대 산출물: `student-next-diagnosis`, `hope-student-bundle` 보고서,
  `plans/STUDENT_NEXT_SINGLE_FACTOR.md`, 정확한 실행명령/실패기록/재개경로.
- 결과 없으면 없다고 쓴다. 코드/통합검사통과를VLA성공이라고쓰지않고,한계와새증거를구분한다.
