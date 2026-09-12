# 과거 상태 — 현재 실행 지시 아님

2026-09-12 문서 정리 전 내용이다. [현재 계획](../../../../RESEARCH_PLAN.md)을 우선한다.

# 현재 작업 상태 — 2026-09-11

Claude Milk 초기 상태 과제 완료: [교차 확인](../../../experiments/2026-09-11/milk-state-crosscheck/README.md).
원 HANDOFF·성공3종해시 유지, 원시 속도 최대/RMS 일치. 새 물리 실행 없이 확인했다.
같은 Claude 명령을 다시 전달하지 않는다. Milk B는 기존A12시범과 계속 분리한다.

## 병렬 실행 추가 — 2026-09-11 23:25 KST


학습 상태 추가 확인: low-noise 학습은 **6,331/6,331 갱신을 완료**했고,
`evaluate-01`로 자동 전환했다. 마지막 가중 train_loss 0.00123087은 실행 기록일 뿐
기존 무가중 손실과 직접 비교하지 않는다. 폐루프 최종 성공 수는 아직 미완료다.

lecun `dexprompt-hope-twelve-v1`: 승인된 fixed7+새5 통합·cache·입력감사·preflight·새후보2개 DINO 검사.
job `/data1/yspark/projects/dexprompt/student-checks/hope-twelve-job-v1/checks`의 status를 확인한다.
`dexprompt-low-noise-report-v1`: 기존 학습 결과를 읽어 최종epoch13과 24관측 비교를 자동 작성.
보고서 `/data1/yspark/projects/dexprompt/student-checks/low-noise-comparison-v1`.
두 작업 모두 새 학습 없음. 진행 중 low-noise 학습은 그대로다. Milk B 병합 금지.
[검사 실행 기록](../../../experiments/2026-09-11/hope-twelve-student/README.md),
[Claude 최대3회 초기상태 검사 전달문](../../FAST_PARALLEL_NEXT_2026-09-11.md).
아래는 이전 시점 기록을 포함한다. 상단 실행 상태를 우선하고 원격 job/status.json으로 확인한다.

## 현재 실행: 낮은 잡음 가중치 새 학습 승인 및 tmux 시작

사용자가 새 학습을 명시적으로 승인했다. lecun GPU1
`dexprompt-low-noise-train-v1` 실행 중(시작22:29).
원격 job `/data1/yspark/projects/dexprompt/student-checks/single-bottle-x0-low-noise-job-v1`.
모델 `/data1/yspark/projects/dexprompt/student-runs/single-bottle-x0-low-noise-v1`.
사전 검사→기록/정지대조군→DINO검사→6,331회학습→epoch1/3/6/9/13 각각3seed 평가→
최종24개teacher관측 진단 자동 연결. 새계수탐색 없음. 학습3시간 상한.
중복 시작하지 말고 job/status.json, 모델/run.json·progress.json을 먼저 확인한다.
실제시작 확인:22:31:30 KST training_started=true, 최초80/6331updates 유한손실.
normalizer/data/cache/DINO/joint계약 A와동일, 설정변경 low_noise_loss뿐.
학습진행중이며 로컬progress는 시작스냅샷이다. 원격상태확인 전 중복실행하지 않는다.
[실행 기록](../../../experiments/2026-09-11/student-low-noise-training/README.md) 참조.

## 최신 완료: 손실 분해와 다음 단일 요인 구현·사전 검사

[결과](../../../experiments/2026-09-11/x0-loss-audit/README.md),
[고정 계획](../../STUDENT_LOW_NOISE_LOSS_2026-09-11.md).
72개 raw예측의 팔 손실 기여75–79%; 첫8팔7–10%. t0손실은 t25/49보다1.44–1.46배.
선택적 low_noise_loss={steps:10,relative_weight:4} 구현. 기본값·샘플러 미변경.
소형 epsilon/x0 train/eval손실·출력 수정 전과bit동일, 관련76테스트 통과.
lecun 별도 `out/student-x0-low-noise-code-v1`에서 실제DINO preflight/dry-run 완료exit0,
유한손실1.1956645, [2,64,27], 모델 불변·갱신0. 결과 회수, tmux 종료.
**새 학습은 미실행**. 실행 승인 후 같은6,331updates 비교 및 epoch13 최종 평가 예정.
기존 frozen code·checkpoint·fixed7/새5·Claude teacher 작업 보존.
과거손실 재집계 시 audit의 frozen-source 사용(현재src에는 새옵션이 추가됨).

## 최신 완료: x0 단일 단계 예측과 생성 경로

[보고서](../../../experiments/2026-09-11/x0-denoising-trace/README.md). lecun GPU1 tmux
`dexprompt-x0-trace-v1`은67.2초·exit0 종료.72개 정답+잡음 직접 예측과24개 생성 경로 완료.
24개 최종 출력 모두 기존16단계 결과와 정확히 일치, 모델 전후 해시 동일.
첫8팔행동 RMSE는 생성 마지막 호출에서24/24 증가; 최초→최종 비교에서는17/24 증가.
정답+작은 잡음 t0에서도17/24는 입력보다 출력 오차가 크다. 낮은 잡음 단계 추정도
불완전하며 단순 샘플링 단계 확대는 해결책이 아니다. 학습 원인은 아직 분리 못 했다.
원시·summary·그림 모두 로컬 회수. 새 학습·물리 실행 없음. 중복 실행하지 않는다.
다음 재학습 설계 전 기존 손실의 timestep/팔·손/chunk offset 기여를 확인할 것.
정답으로 좋은 중간 단계를 골라 sampler 조기 종료를 도입하지 않는다.
기존16단계와 fixed7/새5 보존, Claude teacher 작업과 별개.

## 최신 완료: inference16 대50 비교

[최종 보고](../../../experiments/2026-09-11/student-sampler-steps/README.md).
lecun `dexprompt-sampler50-v1`은81.5초·exit0으로 종료했다. B24개와 비교 결과 회수 완료.
A/B 가중치·관측·라벨·정규화 같음, config 변경은 inference_steps16→50 하나.
**24/24에서 첫 행동 및 첫8팔행동 오차 악화**, 전체64팔행동은8개 개선/16개 악화.
기존16단계 유지. 새 학습·물리 실행0. 중복 실행하지 않는다.
당시 다음 제안이던 x0 t0/25/49 예측과 생성 경로 진단은 위 최신 기록대로 완료했다. 큰 학습이나 HOPE 학습을 자동 시작하지 않는다.
Claude multiccd 비교와 기존 fixed7/새5의 물리 계약은 그대로 보존한다.

## 최신 완료: teacher 관측 행동 생성 진단

[최종 보고](../../../experiments/2026-09-11/teacher-observation-diagnosis/README.md).
고정8시점×3seed=24묶음, lecun GPU1 tmux 작업은 약64초·종료0으로 완료했다.
모델 전후 해시 동일, online/cache 전처리8/8 일치, 기존 진단12개 오차 정확히 재현.
초기5시점 첫 행동 최대 오차0.079–0.170rad, 모두 팔 관절1이 최대다.
초기 행동묶음은 시작에서 앞서고 끝에서 뒤처지는 모양이며 단순 라벨 이동으로
확정할 수 없다. 새 학습·물리 실행0, src 모델 변경0.
원시24개NPZ와 analysis/summary.json·그림을 로컬 보고 폴더에 회수했다.
원격 teacher-observations-v1/job-v1은 보존하고 중복 실행하지 않는다.

이 진단 후 제안했던 **inference_steps16→50 비교**는 위 최신 기록대로 완료했다.
같은8관측×3seed, A는 이번24개 재사용, B24개만 생성하는 상한이다.
checkpoint/data/정규화/성공 기준/limiter를 바꾸지 않는다. 새 학습은 아직 실행하지 않는다.
Claude의 multiccd 물리 비교와 별개로 진행하며 fixed7/새5는 기존 물리 계약대로 보존한다.

## 최신: G3 평가와 새5시범 intake 모두 완료

- [G3 결과](../../../experiments/2026-09-11/student-slew-v1/README.md): 고정 변화량 제한으로 초기
  팔 오차는 감소했지만 완주0/3, 지속 접촉획득0/3. seed1001은908tick 침투 종료를 실패에
  포함했다. 다른 두 실행974tick. evaluation/analysis-v2/comparison.json 로컬 회수 완료.
- [새5시범 결과](../../../experiments/2026-09-11/hope-diversity-student-intake/README.md):
  **5/5, 5,011프레임 통과**. AlphabetSoup1443cca6 centre/yaw+15/X−20,
  Raisins3e8fa799 centre/yaw+15. 전체 물리 재현 오차0, recorded5/5·stationary0/5,
  home reference와 preflight 모두 통과. 첫 preflight DINO 경로 누락은 별도 복구 기록 유지.
- 확정 입력은 `results/2026-09-11/hope-diversity-student-intake/evidence/release.json`.
  원격 개별 데이터/cache는 `/data1/yspark/projects/dexprompt/student-checks/diversity-intake-egl-v1/`.
  기존 fixed7와 미병합. 합계 검사 통과 자료12시범/12,477프레임, 새 학습0.
  이번5에 새 DINO full-model forward는 하지 않았다(기존7만 이전에 실시).
- 자체 로컬/원격 intake 및 report tmux는 종료했다. 중복 실행하지 않는다.
  로컬CPU/원격osmesa 중단 자료와 EGL 최종자료를 구분한다. EGL은device1 사용.
- 이 시점의 다음 작업이었던 teacher 관측 행동 생성 진단은 위 최신 기록대로 완료했다.
  G3 실패를 보고 limiter 재튜닝·새 checkpoint 탐색·큰 학습을 자동 시작하지 않는다.
  기존 모델·fixed7·Claude teacher 산출물 보존. 새 학습은 아직 실행하지 않는다.

아래 내용은 이전 시점 기록이다.

## 이전 기록: Claude C0–C3 교차 확인 완료, G3 평가 실행 중

사용자가 Claude 완료 보고 확인과 우리 다음 작업 시작을 지시했다.
`results/2026-09-11/hope-diversity-crosscheck/README.md` 참조: 새 승인 6건/고유 후보3개,
재현5건, 24회 episode/model/초기상태 해시 일치. 저장신호 student guard 6/6 위반 없음,
전체 시각 인수는 미실행. 중앙 실패 실제 phase는 close8/lift1이며 보고의 carry4와 구별.

lecun tmux `dexprompt-x0-slew-v1`에서 G3 평가를 시작했다. GPU0 UUID 사용, 고정3seed,
최대974tick, 1시간 상한, 새 학습 없음. 원격 상태는
`/data1/yspark/projects/dexprompt/student-checks/x0-slew-job-v1/status.json`, 평가 출력은
같은 student-checks의 `x0-slew-v1`. 중복 시작하지 말고 status/PID/출력부터 확인한다.
`results/2026-09-11/student-slew-v1/README.md`의 결과 회수 규칙을 따른다.
아래 G3 미실행은 이 시작 이전 기록이다.

## GPT-5.6 G0→G3 완료 — 아래 진행 중 기록보다 우선

`plans/GPT56_STUDENT_HANDOFF_2026-09-11.md`의 student 범위를 완료했다. 상세 결과는
`results/2026-09-11/student-next-diagnosis/README.md`, 기계 판독 인계는 같은 폴더의
`bundle-release.json`이다. **새 학습은 시작하지 않았다.**

- G1: 단일 병 x0 세 실행은 원본과 같은 최초 8개 입력 배열을 받았지만 첫 팔 명령부터
  0.158–0.170 rad 틀렸다. 첫 64 tick의 재계획 경계 명령 변화는 구간 내부의
  26.6–33.3배다. 최초 원인은 폐루프 렌더/물리 재생보다 정책의 action 생성에 있다.
- G2: 고정 HOPE 7시범/7,466프레임/3물체/7배치/3후보/2유형을
  `/data1/yspark/sim/icra/hope-student-bundle-v1`에 통합했다. manifest SHA256은
  `d9e98792ad218562…`다. 기록 행동 7/7 성공, 정지 7/7 비성공을 로컬과 lecun에서 확인했다.
- lecun: `/data1/yspark/projects/dexprompt/student-data/hope-bundle-v1`, cache는
  `/data1/yspark/projects/dexprompt/student-cache/hope-bundle-v1`. `.partial` 전후 156파일,
  1,653,463,986바이트 해시가 같다. 물체별 실제 DINO batch 3개 모두 `[2,64,27]`,
  유한 loss/action, 모델 불변, backward/optimizer 0이다.
- 첫 원격 DINO 검사에서 cache에 없는 `phase`를 읽는 helper 오류가 났다. source NPZ를
  읽고 네 CPU 성공 단계를 검증한 뒤 재개하도록 고쳤다. 데이터·모델·teacher는 불변이다.
- G3: `plans/STUDENT_NEXT_SINGLE_FACTOR.md`. 기존 x0 평가에 teacher 범위의 고정 관절별
  명령 변화 제한만 적용하는 무학습 3회 비교다. 구현·회귀 검사는 완료했지만 실행하지 않았다.
- 전체 572 tests 통과, 새·수정 파일 Ruff 통과. 원격 `dexprompt-hope-bundle-check`는
  완료되어 종료됐다.

다음 실행은 G3의 고정 평가 여부를 결정하는 것이다. 결과를 보고 limiter를 조정하거나
checkpoint를 바꾸지 않는다. HOPE 묶음은 train-only 진단 자료이므로 독립 validation/test,
HOPE 정책 학습·폐루프, point/language와 clutter는 여전히 남는다.

Claude의 C0→C3 teacher 작업과 출력은 별도 소유이며 이번 작업에서 수정·중지하지 않았다.
`plans/HANDOFF_NEXT_2026-09-11.md`의 이전 분담 기록은 실행 조건의 근거로 보존한다.

## 17시대 최종 통합 — 아래 진행 중 기록보다 우선

`results/2026-09-11/consolidated-status/README.md`가 최신 통합 결과다.
- Claude HOPE 완료: 중앙 3/36 + 확대 5/12 = 유효 48조건 accepted 8, 동일 재현 8.
  격리된 Raisins 실제 4회 추가 기록이 있어 총 물리 비용을 48로 쓰지 않는다.
- Student 인수 **7건/7466프레임/3물체/7배치/3후보/2유형**, 초기 정책 입력7개.
  `out/hope-student-pool-final.json`; AlphabetSoup 중앙만 계약 차이로 보류.
- x0 6331 updates 완료, epoch1/3/6/9/13 모두0/3. 최종 접촉 획득0, 상승약1mm이하.
- Peaches lecun 실제DINO 검사까지완료, backward/optimizer0. 새HOPE학습없음.
- 누락된 audit_student_failure.py만 lecun격리사본에추가해 CPU감사 재개:
  tmux `dexprompt-x0-audit`, `out/x0-command-audit-recovery-v1`.
  이전10시간후속은 해당파일누락으로failed였으며 성공으로덮지않음.
  복구감사완료: 세실행974tick 물리신호차이0,정규화/소스/관절순서/전처리/시범상태정렬통과.
  결과는 consolidated-status/x0-command-audit.json. 새학습없음.
- 다음은 감사결과확인,7건자료통합/전송,단일과제실패원인에맞춘다음실험설계.

## 04:50 최신 상태 — 아래 시간대 기록보다 우선

사용자가 승인한 10시간 작업은 **12:41:16 KST까지 계속 진행 중**이다. 새 학습을
추가하지 않고 기존 x0 평가, 그 결과에 따른 무학습 진단, 새 HOPE 데이터 인수를 한다.

- lecun `dexprompt-x0-bc`: 학습 **6,331 updates 완료**, 사전 지정 평가 epoch 9 진행.
  epoch 1/3/6은 각각 0/3. 최종 epoch 13 결과를 기다린다. 중간 결과로 모델을 고르지 않는다.
  `dexprompt-x0-follow`는 최종 3회 결과에 따른 기존 분기를 대기 중이다.
- Peaches 중앙: 실제 12회 중 1회 완주, controller 재실행 해시 동일. 1,006 tick
  시각 변환·목표 기억·기록 명령/정지 대조군·preflight 모두 통과했다.
  `out/hope-student-readiness-0421.json`: 새 데이터 ready 1개. 독립 test나 정책 성공이 아니다.
- Peaches 데이터 14파일/235 MB를 lecun에 전송해 모든 SHA를 대조했다.
  `dexprompt-hope-peaches-check`: CPU 검사를 마치고 x0와 후속 작업 종료를 기다린다.
  이후 실제 DINO forward만 검사하며 backward/optimizer 호출은 금지한다.
  원격 data/checks: `/data1/yspark/projects/dexprompt/student-{data,checks}/hope-pilot-v1/Peaches`.
- AlphabetSoup는 teacher 성공이지만 기존 student 획득 전 이동 제한 때문에 인수 보류.
  정확한 원본 재생에서도 tick 621에 걸린다. 제어·성공 기준·현재 학습 평가를 바꾸지 않는다.
- Claude `icra-hope4`는 Raisins 생성 중이다. v4 budget 집계는 현재 24회 물리 실행과
  2회 재현으로 일치한다. 과거 0-physics 기록 중복 집계는 현재 실행에 대해 해결됐다.
- 로컬 중앙 인수: `dexprompt-visual-intake-v2`, `out/hope-student-intake-v2`.
  추가 배치 인수: `dexprompt-placement-intake`, `out/hope-placement-intake-v1`.
  후자는 이미 계획된 네 배치의 실제 실행+동일 재현을 기다린 뒤 순차 CPU 검사한다.
  archive 신호 사전 검사는 거절만 할 수 있으며, 통과하면 전체 시각 재생과 대조군이 필요하다.
- 읽기 감시 exec session 26579는 45분 상한(약 05:01까지), 기록은
  `out/ten-hour-monitor.jsonl`. 예전 35182/44179 등은 종료됐으므로 재사용하지 않는다.
- 테스트: 확장한 인수 감시와 재현 귀속 관련 17개 통과. 실행 중인 중앙 감시기는
  이전 코드를 이미 읽었으므로 새 확장 옵션 때문에 재시작하지 않았다.

아래는 이전 시점 기록이며 현재 진행 여부는 위와 실제 status 파일을 먼저 확인한다.

## 03:55 중요 갱신 — 인수 계약 불일치
- AlphabetSoup 실제12회1성공, controller 재현해시 동일. 1192tick visual변환 물리오차0, goal ref0/387 검사통과.
- 그러나 student controls는 tick621 close 중 approach_displacement로실패. 기존guard_v2는5군연속0.5sec까지접근, teacher는close를변위제한에서제외. 정확한시범복제도막힘. results/2026-09-11/student-acquisition-contract 참조. 학생데이터채택보류, 물리/학습/평가기준미변경.
- 원본병3+캔1은같은guard감사위반없음. lecun x0소스/계약보존. 새HOPE기술DINO검사는별도진단으로만가능, approveddata로부르지말것.
- wrapper중복budget실증: 옛0physics stage가새executed폴더를재읽어12실행을24로셈. out/hope-pilot/CODEX_REVIEW.md에Claude인계, 공유코드프로세스미변경.
- monitor exec35182운영중(45분상한), 이전44179종료. localintakev2계속다른물체대기. release CLI scripts/report_hope_student_readiness.py 추가6tests.


## 03:28 최신 추가
- AlphabetSoup 최초12개는 입력계약거절, 실제물리실행0. Claude가 flat scene identity와 wrapper v3 수정/재개 중. Codex 수정본으로 동일12개 checked_candidate 통과 확인; 공유 collector 미수정.
- 인수 v1 정상 STOP 종료. 현재 tmux dexprompt-visual-intake-v2, out/hope-student-intake-v2. 최신execute/실제episode확인, 수정manifest 재인수. 새테스트포함관련36통과.
- lecun x0 3930/6331, 약45분 남음. follow는 자동평가 종료 대기. monitor session44179 종료됨; 다시poll하지말것.


### 10시간 작업 추가 메모(계속 진행할 것)
- out/monitor-ten-hour-pass.py를SSH 읽기감시로 실행 중. exec session44179,30분상한,
  55초마다 out/ten-hour-monitor.jsonl 기록. student train 종료/첫intake시 조기반환.
- 마지막 확인x0 3310/6331,잔여약57분. 현재시각약03:17 KST,10시간마감12:41:16.
- Claude가wrapper v2와out/hope-pilot/run-config.json/logs/quarantine을추가해안전경계재개.
  AlphabetSoup 두유형×seed0/1 완료,Large Diameter seed2정련중. 아직handoff없음.
- HOPE근접형상감사완료: results/2026-09-11/hope-geometry-groups.21near pairs,
  dev7개+동그룹포함총14개는새형상test로바로세지않도록목록만작성. 최종split아님.
- 도구연결후대기중에도실제결과가나올때까지진행. 새학습은lecun만,현재학습/기준유지.


## 최우선: 10시간 무승인 자율 작업 진행 중

사용자가 별도 승인 없이 계속 작업을 명시. 작업창09-11 02:41:16–12:41:16 KST.
`results/2026-09-11/ten-hour-codex/README.md` 참고. 끝났다고 응답하지 말고 결과까지 계속.
- lecun: x0기존학습 tmux `dexprompt-x0-bc`, 새무학습후속 `dexprompt-x0-follow`.
  후속은최종epoch13 완주≥1이면paired10seed(x0/epsilon),0이면x0진단+명령물리재생.
  out/ten-hour-student-followup-v1. deadline1789098076. source체크우회없음.
- 로컬: `dexprompt-visual-intake` 감시작업. out/hope-pilot 각물체execute완료후첫accepted1건씩
  visual변환/cache/audit/controls/preflight. out/hope-student-intake-v1은/data1저장symlink.
  Claude teacher재실행재현은별도확인필요. teacher출력/시도예산/제어변경하지않음.
- x0대응진단/완료상태분기/인수도구 신규21tests 및관련검사통과. 모델/학습코드추가수정없음.
- 캔1시범과병캔4시범인수검사통과. 4시범의고유초기정책입력2개,병3개첫chunk다름.
- root여유1.8GB를발견해우리cache3개를/data1/student-caches로이전(91파일hash동일),
  기존경로symlink유지. root여유약6.8GB. 캐시원본내용/teacher시범/타인파일변경없음.


## 최신 — 실행 중 HOPE 파일럿 교차 확인

`results/2026-09-11/hope-pilot-crosscheck/README.md`: 정련54/정적39 확인.
예측111 보고는 원시파일/로그140과 불일치. 큰그릇정련10의 역사 결과도 정정 필요.
wrapper가 placement/align/refine 실패를 모두 차단하지 않으며 timeout 하위worker/전체로그
처리 보완 필요. Claude 코드/프로세스는 건드리지 않았다. 현재 정상 seed0결과 보존.
10시간 계획의 후보 순환 제안은 기존 hash상위12 규칙과 달라 기존 규칙 유지로 정정.
다음은 물체당6정련 후 동적12회, 성공 재현이며 18정련 전체까지 첫실행을 기다리지 않음.


## 최신 분담 계획 — 다음 10시간

`plans/TEN_HOUR_SPLIT_2026-09-11.md`에 전체 상태·10시간 분담·분기·Claude 전달문 고정.
Claude: HOPE 중앙3물체(총36회)→성공 재현→조건부 네 배치 확대. 전체 새 시도48+재현15 상한.
Codex: 현재 x0 학습/자동 평가, 결과에 따른 추가10seed 비교 또는 무학습 실패 진단,
새 시범의 visual-v2 인수/데이터 검사. HOPE 혼합 학습은 이번에 추가하지 않는다.
상태 확인 시 x0는790/6331 updates, 유한 loss0.01440, 학습 잔여 약1h45m+평가.
새 실험/자동 평가 분기를 이번 계획 작성 중 추가 실행하지 않았으며, Claude에게 메시지를 보내지도 않았다.
상세 명령은 새 계획의 전달문을 사용자가 전달한다. 전체 진행 추정35% 유지.


## 최신 — x0 비교 구현 완료, lecun tmux 실행 시작

사용자 "우리도 진행하자" 및 "아니야 lecun 그냥 먹자"에 따라 공유 GPU 실행.
`results/2026-09-11/single-student-x0/README.md`가 최신 실행 계획이다.
- x0 학습 목표+sampler 함께 구현, 생략 시 epsilon 유지. 관련 82 tests, Ruff 통과.
- lecun 원본 checkout 보존. `out/student-x0-code-v1`에 별도 코드 사본.
- tmux `dexprompt-x0-bc`, GPU 0 공유, train/full-model dry-run 메모리 상한 85%.
- 기록 `out/single-bottle-x0-job-v1`, 모델 `/data1/yspark/projects/dexprompt/student-runs/single-bottle-x0-v1`.
- 같은 974 tick 단일 병 데이터, seed42, batch2, 13epoch/6331 updates.
  사전 검사·대조군·실제 DINO dry-run 통과 후 학습, epoch1/3/6/9/13×3seed 자동 평가.
- 사전 검증·대조군·실제 DINO 검사 통과 후 실제 60/6331 updates, 유한 loss 0.19183 확인.
  run.json training_started=true, tmux 학습 진행 중. startup.json에 시작 증거 보관.
  아래의 "x0 구현/학습 전"은 이전 기록이다. HOPE 수집은 클로드 담당.


## 최신 — HOPE 초기 장면 준비 및 단일 정책 무학습 진단 완료

사용자 "응 우리쪽도 진행해"에 따라 두 작업을 완료했다.
`results/2026-09-11/hope-pilot/README.md`와 `single-student-diagnosis/README.md` 참조.

- HOPE 6개 geometry 생성 완료: `out/hope-pilot-v1/objects/`.
- `object_appearance: texture_v1` opt-in, OBJ/MTL/UV 검증과 JPEG→PNG 메모리 변환.
  `configs/scene_hope_textured.yaml`에서만 적용한다. 기존 solid 기본값 유지.
- `scene.object_yaw_degrees`가 모든 builder 경로에 적용된다. 해당 장면을 쓸 때
  Recorder episode yaw는 0으로 둔다(추가 yaw와 중복 적용 금지).
- 30개 초기 장면 정합·렌더 검사. AlphabetSoup/Peaches/Raisins만 각각 5배치 통과.
  Mustard/BBQSauce/ChocolatePudding은 1.2초 후 회전 속도 미달, 중앙에서 4초까지
  확인해도 안정되지 않았다. 한계를 바꿔 통과시키지 않았다.
- 6개 중앙 조건 solid/texture 600-step physics 동일. Dexonomy 실제 물체 importer와
  eval 밀도 규칙으로 질량/관성 비교, 6개 모두 실행기와 일치.
- `out/hope-pilot-v1/ready-objects/`에는 통과 세 물체만 링크했다.
  `generation-plan.json`에 15 eligible/15 pending 배치, scene/mesh/template 해시,
  유형 2개씩·align seed 0/1/2·refine 400·중앙 동적 실행 12회/물체 상한 기록.
  새 파지 시범은 아직 없다. 다음은 세 물체 중앙 조건의 teacher 파지 수율 확인.
- lecun GPU 0에서 최종 epoch 13을 4개 고정 학습 관측으로 무학습 진단했다.
  낮은 잡음은 0.0013–0.0038 rad RMSE로 복원하지만 순수 잡음 생성의 큰 오차는 남음.
  sampler 첫 고잡음 단계에서 정규화 좌표 약 98%가 clipping, 팔에서도 확인.
  이것만으로 원인 확정/기존 코드 오류 판정은 하지 않는다. x0 목표 비교는 제안만.
- 관련 테스트 72개(신규 12개)·변경 코드 Ruff·diff check 통과. 새 학습 없음.
  로컬 생성/검증 프로세스와 lecun 진단 tmux는 종료됐다.

아래는 이전 확인 시점의 기록이다.

## 최신 — HOPE 자산 점검과 lecun 종료 확인 (09-11)

사용자 요청: 그릇/망치 스케일을 확인하고 HOPE 기반 여러 물체·배치 수집을 준비한다.
`results/2026-09-10/hope-data-readiness/README.md`가 새 데이터 수집 후보·준비 조건이다.
공식 full/eval zip을 다운로드·체크섬 검증했고 28개 치수를 전수 측정했다.
full→m은 0.001, eval→m은 0.01이다. 기존 GraspGenX 3개는 full과 바이트 동일하며
기존 자산의 0.001 변환은 맞았다. source는 `out/hope-audit/source/`에 별도 보관한다.
3개 기존 모델의 물체 material id는 전부 -1: 새 시각 데이터 전 텍스처 연결 필요.
신규 25개는 충돌 자산/놓임 검증 전이며, 새 teacher 생성·파지 실행은 하지 않았다.

lecun `out/single-bottle-job-v1/status.json`: completed, running false.
단일 시범 학습 6,331회 갱신 후 epoch 1/3/6/9/13 각각 완주 0/3이다.
최종 13도 0/3. `results/2026-09-10/single-bottle-student/completed-summary.json` 참조.
raw 결과는 `out/single-bottle-job-v1-server/results.json`에 복사했다. 추가 학습 없음.
HOPE를 늘리면 이 전이 문제가 자동 해결된다고 가정하지 않는다.
아래의 학습 진행 중 기록은 이전 확인 시점의 기록이다.

## bowl-drop 진단 후속 검토

`results/2026-09-10/bowl-drop-crosscheck/README.md`를 확인한다.
hold 중 접촉 상실은 지지되지만, 자세 최대 오차로 손가락 2/4 닫기 부족을
원인으로 확정할 수 없다. 회전 통계는 첫 tick 기준 scalar 값이므로 lift 기준의
회전 행렬 차로 재계산이 필요하다. squeeze 대체 실험은 아직 실행하지 않았으며
전환·목표 도달·품질 게이트·동일 팔 스케줄과 해석 범위를 먼저 고정해야 한다.
lecun 학습은 확인 시점 3,850/6,331 updates로 정상 진행 중이었다. 자동 평가 연결 유지.

## 최신: lecun 단일 시범 학습 파이프라인 실행

Claude의 그릇 2×2 완료 기록도 교차 확인했다. 큰 그릇×matcher 4에서 정련/실행 10,
완주 0. lift_timeout 8개 모두 최종 접촉 0/5·테이블 높이 복귀를 원시 신호로 확인했다.
`results/2026-09-10/bowl-scale-crosscheck/README.md`의 완료 후 절 참조.
`build_expansion_manifest.py`의 이전 출력 재입력/빈 결과의 오래된 출력 잔존을 수정했다.
실험 파일은 보존했고 관련 8개 테스트 통과. 새 teacher 실험은 시작하지 않았다.

사용자가 다음 작업 실행을 지시해 `dexprompt-single-bc` tmux를 시작했다.
데이터/캐시/preflight·기록 행동 대조군을 통과하고 **학습 갱신 100회, 유한 손실**을 확인했다.
학습이 끝나면 사전 지정 평가가 자동으로 이어진다.
아래 과거 절의 "새 학습 미승인/준비 전" 상태보다 이 절이 우선한다.

- 서버: lecun `/home/yspark/projects/DexPrompt`, GPU 0 UUID
  `GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035`만 사용. 기존 서버에서는 학습하지 않는다.
- 상태: `out/single-bottle-job-v1/status.json`, `out/single-bottle-launch-v1.log`.
- 학습: `/data1/yspark/projects/dexprompt/student-runs/single-bottle-v1`의
  `run.json`(`training_started`), `progress.json`, `metrics.tsv`.
- 상한 13 epoch / 6,331 updates / 학습 3시간. 새 출력이며 기존 결과 보존.
- 학습 완료 후 epoch 1/3/6/9/13 × seed 1000–1002 평가. 최종 epoch는 13으로 사전 고정.
- 실패/timeout이면 다음 단계 중지, checkpoint 보존. 네트워크 단절만으로 다시 실행하지 않는다.
- 보고서: `results/2026-09-10/single-bottle-student/README.md`.

## 실행 장소와 최신 teacher 검토

사용자가 **학습은 lecun 서버에서만 실행**하도록 지정했다. tmux 사용을 유지한다.
단일 병 시범의 설정/예산은 `plans/SINGLE_DEMO_STUDENT_LECUN.md`에 있다.
13 epoch, 약 6,331 updates의 준비 설정이며 데이터/캐시/래퍼 준비와 학습은 아직 안 했다.
Claude의 그릇 scale 보조 실험은 진행 중에 원시 기록을 확인했다:
`results/2026-09-10/bowl-scale-crosscheck/README.md`.
matcher 4에서 큰 그릇은 정련 파일 10개를 얻었지만 그 시점에 완주 평가는 진행 중이었다.
init loss/collision의 0, 변형/원래 후보 분모, 래퍼 오류 처리 보완이 필요하다.

## 최신 상태: student 초기 실패 감사 완료

이 절이 아래 이전/대기 상태 기록보다 우선한다. lecun 환경 설치와 12회 평가는
이미 완료됐다(완주 0). 추가 무학습 진단도 완료했으며 새 학습은 시작하지 않았다.
원래 checkpoint·데이터·학습/전처리 소스·teacher 제어/기준은 그대로다.

- 보고서: `results/2026-09-10/student-failure-audit/README.md`.
- CPU 물리 재생 12개 모두 원래 신호와 정확히 같다. 같은 seed의 병 3개 기록은
  궤적이 완전히 중복이며, 초기 조건은 병/캔 2개다(병 예산 971/974 tick 차이는 별도).
- normalizer·27 joint mapping·시간 정렬·16개 상태의 train/inference 입력 tensor 일치.
- 첫 명령의 최대 시범 오차: 병 0.71–1.45 rad, 캔 0.82–1.69 rad.
- lecun GPU 0에서 16→50회 추론 비교도 큰 오차를 해소하지 못했다.
  `dexprompt-diagnosis` 종료 코드 0. `out/student-denoising-audit-exit.txt`와 로그 참조.
- 관련 테스트 52개 통과. 전체 코드에 오류가 없다는 주장은 하지 않는다.
- 다음은 고정 병 시범 하나의 제한된 시각 학습 진단 계획을 확정하는 것.
  현재 지시는 진단까지이며 새 학습을 자동으로 시작하지 않는다.
- Claude teacher 작업은 먼저 침투 장면의 단계와 bowl scale을 확인하는 최신 대화가
  우선한다. 새 유형 확대를 곧바로 실행하지 않는다.

## 새 서버로 이동 준비

사용자가 `https://github.com/teamROBI/DexPrompt.git` 업로드와 3090 서버 이전 준비를 요청했다.
설치·전송·검사·tmux 실행 명령은 [SERVER_3090.md](../../../../docs/lecun.md)에 고정했다.
전송 묶음은 `/data1/yspark/sim/icra/student-transfer/bottle-can-eval-v1.tar`이며
SHA256은 `66ffda013302d5816d8d2d9ac314f9ef71725d02b5237c793e4e0668047bf1f7`이다.
기존 서버에서 새 학습이나 학습 정책 평가를 재시작하지 않는다.
새 서버의 다음 작업은 최종 epoch_0002 체크포인트의 4개 과제 × 3회 평가다.
GitHub push는 아직 미완료다: 서버 Git 인증이 없고 연결 앱의 저장소 권한도 `push: false`다.
업로드용 별도 Git 저장소는 `out/github-publish/DexPrompt`, 오프라인 clone용 묶음은
`out/github-publish/DexPrompt.bundle`이다. 쓰기 인증 후 별도 저장소에서 `git push -u origin main`을 실행한다.
원래 프로젝트의 `.git`/index는 변경하지 않았다.


### lecun 전송 완료

사용자가 `/data1/yspark/projects/dexprompt`로 대용량 파일 전송을 요청했다.
SSH: `yspark@147.47.200.171`, port `19423` (`lecun`).
2026-09-10 접속은 됐지만 `Permission denied (publickey)`로 인증 실패했다.
현재 서버에는 기존 클라이언트 SSH 키나 agent가 없어 전송용 키를
`out/server-transfer/lecun/id_ed25519`에 만들었다. 비밀키는 출력하거나 전송하지 않는다.
사용자에게 같은 경로의 `.pub` 내용을 lecun `~/.ssh/authorized_keys`에 등록하도록 안내했다.
**공개키 등록 후 인증 성공, tmux `dexprompt-transfer-lecun`에서 전송과 원격 SHA256 검증을 완료했다(exit 0).**
목적지 `/data1/yspark/projects/dexprompt`에 tar 6,021,109,760 bytes와 `.sha256`가 있다.
원격 결과: `bottle-can-eval-v1.tar: OK`. 전송 약 51초, 평균 111MB/s.
아직 압축 해제나 환경 설치·학습·평가는 하지 않았다.
lecun의 /data1 여유 공간은 시작 시 592GB였다. 실행 스크립트는
`bash /home/yspark/projects/ICRA/out/server-transfer/lecun/transfer.sh`.
재개 시 먼저 `out/server-transfer/lecun/exit-code.txt`와 `transfer.log`, tmux 세션을 확인한다.
중복 전송을 시작하지 않는다.
스크립트는 tar·sha256만 지정 폴더로 전송하고 원격 해시를 검증한다. 압축 해제나 학습은 하지 않는다.
로그·시작·종료·exit-code는 `out/server-transfer/lecun/`에 기록한다.

## 기존 서버의 학습·평가 상태

사용자가 공용 서버의 다른 작업을 배려하며 시각 student BC 학습을 tmux에서
실행하도록 승인했다. **학습은 15:47:09 KST 종료 코드 0으로 완료됐다. 중복 실행하지 않는다.**

- 사용했던 tmux: `icra-student-bc`, 학습 종료 후 세션이 사라질 수 있다.
- 물리 GPU 0 한 장, allocator 50% 제한, CPU 스레드 1, nice 10, ionice 3,
  계산 시간과 같은 길이의 스텝 휴지 시간.
- 출력: `/data1/yspark/sim/icra/student-runs/bottle-can-bc-v1`.
- 로그: `out/student-bottle-can-launch-v1/train.log`.
- 종료 코드: `out/student-bottle-can-launch-v1/exit-code.txt`(실행 중이면 없음).
- batch 2, epoch 3, 총 6,255 updates 완료. epoch 손실 0.1055 → 0.0286 → 0.0172.
- epoch_0000.pt~epoch_0002.pt 세 파일 확인, 학습 소스 해시 불일치 0.
- 최종 checkpoint의 CPU 폐루프 평가를 시작했으나 사용자 요청으로 중지했다. 최종 성공률은 미집계다.
- 설정·데이터·학습 소스는 학습 도중 바꾸지 않는다.
- 클로드의 teacher 접촉 진단은 `icra-diag`에서 별도로 실행 중일 수 있다.
- `jokim` 계정의 기존 Python/GPU 프로세스는 다른 프로젝트이며 건드리지 않는다.

먼저 위 로그와 progress.json, tmux 세션 상태를 읽는다. 네트워크 단절만으로
학습을 재시작하지 않는다. 학습 완료 후 다음 작업은 `epoch_0002.pt`의
병·캔 4시범 × 3회 폐루프 평가다. 학습 일반화가 아닌 train-task 진단으로 보고한다.

상세 기록: [학습 실행 보고서](../../../experiments/2026-09-10/visual-student-bc-v1/README.md).

클로드의 접촉 진단도 완료됐다. 다음 조건 결정 전
`results/2026-09-10/contact-diagnosis-crosscheck/README.md`를 읽는다.
초기화 비교 시 align seed만 바꾸면 템플릿·접촉점도 바뀌므로 이를 별도 고정해야 한다.

## 폐루프 평가 시작 — 2026-09-10 16:18 KST

CPU 평가를 시작했으나 이후 사용자가 오래 걸리면 나중에 하자고 요청하여
**`icra-student-eval`에 Ctrl-C를 전달해 평가를 중지했다. 자동으로 재시작하지 않는다.**
다른 사용자 GPU 학습이 네 장 모두 약 99% 점유하여 CPU/OSMesa 1스레드로 실행했다.
최종 epoch_0002.pt, 병 3+캔 1 시범, 각 seed 1000–1002(12회), 학습 과제 진단이다.
- 출력: `/data1/yspark/sim/icra/student-runs/bottle-can-bc-v1-eval-cpu`.
- 로그: `out/student-bottle-can-eval-launch-v1/evaluate.log`, 시범별 로그는 출력 폴더.
- 종료코드: `out/student-bottle-can-eval-launch-v1/exit-code.txt`.
- 모델·기준 변경 없이 결과를 확인한다. 중복 실행하지 않는다.
- 상세: `results/2026-09-10/visual-student-eval-v1/README.md`.

## 평가 보류 — 사용자 최신 지시

GPU가 모두 사용 중인 동안 CPU 평가는 오래 걸려 나중으로 미뤘다.
중간 파일(initial_observation 및 replay_000.npz)은 보존하되 전체 평가 결과로
집계하지 않는다. 학습 checkpoint 3개는 그대로다. GPU 여유가 생기고 사용자가
다시 진행을 요청하면 새 평가 출력 폴더를 사용한다. 기존 폴더를 덮어쓰지 않는다.

## Teacher 초기화 진단 후속 검토

클로드의 루트 초기화 실험은 음성 0/30, 양성 9/10.
`results/2026-09-10/init-sensitivity-crosscheck/README.md`에 저장 목표 검사 오류와
현재 original 기준 접촉 상실 5개 정정을 기록했다. 관련 스크립트 2개 수정 및
회귀 테스트 5개 추가, 40개 실제 파일 목표 보존 확인. 새 실험은 실행하지 않았다.
Student 평가 보류 및 다른 서버 이전 보류라는 사용자 지시는 그대로 유효하다.

## 손가락 초기값 진단 완료·검토

Claude의 finger5-flexion: 음성 실행 0/28, 범위초과 미실행 2/30, 양성 9/10.
Codex가 실행 입력 38개·원본 8개·종료 기록과 기존 기준선을 직접 대조했고
주 결과 불일치 없음. 관련 22개 테스트 통과. 이 여섯 후보의 추가 미세조정은 보류.
다음 선택은 유형 변경이며 아직 실행하지 않았다. 유형 비교 시 반복 예산도 동일하게
고정해야 한다. 상세: `results/2026-09-10/finger5-crosscheck/README.md`.
Student 평가 및 서버 이전은 계속 보류하며 자동 재시작하지 않는다.

### lecun 환경 설정·평가 착수

사용자가 `/home/yspark/projects/DexPrompt`에 clone했고 SSH로 환경 설정·tmux 실험을 승인했다.
clone 커밋은 `5637da3`, 초기 작업트리는 깨끗했다. GPU 0만 유휴, GPU 1–5는 다른 작업 중이다.
`dexprompt-setup` tmux에서 설치·해제·모델 로드·CPU 대조군 검사를 진행한다.
최초 `ensurepip` 누락은 프로젝트 가상환경의 공식 PyPA pip 부트스트랩으로 보완했다.
`out/server-setup-v1/setup.log`, `exit-code.txt`를 확인한다. 설치 실패 기록은 별도 파일로 보존했다.
평가 실행용 스크립트는 로컬 `out/server-transfer/lecun/evaluate-remote.sh` 및 원격
`out/server-evaluation-v1/evaluate.sh`에 있다. 원격 `dexprompt-eval-queue` tmux를 이미 시작했고
설정 성공 후 smoke → 본 평가가 자동으로 이어진다. 대기열을 중복 시작하지 않는다.
설정 통과 후 GPU 0에서 smoke 1회 → 고정 4개 과제 × 3회로 진행한다. 새 학습은 하지 않는다.

새 서버 실행 기록: [lecun-student-eval](../../../experiments/2026-09-10/lecun-student-eval/README.md).

lecun 환경 설치·모델 로드·물리 대조군 4개 모두 통과(exit 0).
`dexprompt-smoke`에서 GPU 0의 CUDA/EGL 사전 검사가 통과했고 첫 정책 평가가 실행 중이다.
원격 `out/eval3090-smoke-v1/status.json`은 running (시작 2026-09-10 21:09:23 KST).
smoke의 실행 오류가 없으면 `dexprompt-eval-queue`가 `dexprompt-final`을 자동으로 시작한다.

lecun smoke는 정상 실행 종료(exit 0), 결과 0/1 (`penetration`, 17 tick).
대기열의 `quality_` 접두사 가정 오류를 실제 quality_violation 반환값 6종으로 수정했다.
`dexprompt-eval-queue`를 `evaluate.sh --after-smoke`로 재개해 본 평가만 이어가고 있다.
smoke를 재실행하지 않았다. 새 서버의 `out/server-evaluation-v1/launch.log`와
`out/eval3090-final-v1/status.json`, `results/summary.json`을 먼저 확인한다.

**lecun 본 평가 실행 확인:** 2026-09-10 21:14:13 KST `dexprompt-final` tmux 시작,
GPU 0 CUDA/EGL 통과, `out/eval3090-final-v1/status.json` running.
`dexprompt-eval-queue`는 완료 상태를 기다린다. 12회 평가를 중복 실행하지 않는다.

**최신: lecun 본 평가 완료(21:21:12 KST, exit 0), 12개 기록에서 완주 0.**
원시 결과를 로컬 `out/lecun-student-eval-v1/`로 회수했다.
보고서 `results/2026-09-10/lecun-student-eval/`의 final.json/status.json을 확인한다.
병 3시범의 초기 입력 배열이 동일함을 확인했으므로 독립 과제 분모를 따로 감사해야 한다.
다음은 `plans/NEXT_WORK_2026-09-10.md`: Codex=P5 초기 행동·평가 계약 감사,
Claude=P3 정식 유형 변경의 제한된 새 시범 확보. 이 문서로 추가 학습을 시작하지 않는다.
