<!-- research-scope-2026-09-12 -->
> **날짜별 실험 계획 기록:** 현재 목표·우선순위는 [연구 정의](../../RESEARCH_PLAN.md)와
> [현재 작업서](../../plans/CURRENT_WORK.md)가 우선한다. 아래 실행 지시·P번호·미실행/진행 상태는 작성 당시 것이다.
> 고정 후보 접근 PPO는 형상 기반 선정/적응 RL 전체와 다르다. 부위 지시는 필수 범위가 아니다.
> 완료된 작업을 이 문서만 보고 재시작하지 않는다. 기존 승인 및 원본 보존은 유지한다.

# 단일 병 x0 낮은 잡음 손실 가중치 비교 — 실행 승인됨

## 질문과 상태

균등 x0 손실에서 낮은 잡음 단계의 상대 가중치만 높이면, 학습 관측에서 나타난
초기 팔 행동 오류와 폐루프 실패가 개선되는가? 원인 확정이 아닌 단일 요인 가설 실험이다.
[분해·구현·사전 검사](../experiments/2026-09-11/x0-loss-audit/README.md) 완료.
사용자가 “새 학습 시작하자”로 실행을 승인했다. lecun tmux
`dexprompt-low-noise-train-v1`에서 사전 검사→학습→고정 평가를 시작했다.
실제 optimizer 시작 여부는 run.json의 training_started와 progress.json으로 확인한다.

## A/B 계약

- A: 기존 단일 병 x0 seed42,13epochs/6,331updates, 최종 epoch13 출력 파일
  `epoch_0012.pt`, SHA `a5100b38f9c1fb64…`. 기존 완주0/3 자료를 보존한다.
- B: 같은 초기 seed42에서 처음부터 학습한다. 기존 checkpoint 미세조정이 아니다.
  같은 단일 병974프레임·normalizer·DINO·batch2·학습률·optimizer·schedule·noise assignment·
  dropout·horizon64/execute8·inference16·13epochs/6,331updates를 사용한다.
- 유일한 설정 차이: `low_noise_loss={steps:10,relative_weight:4}`.
  균등 timestep 추첨은 유지하고 손실만 t0–9 2.5배 / t10–49 0.625배 한다.
  팔/손 및 offset 가중치, 마찰·제어·goal memory·mask·물리 성공 기준은 유지한다.
- 10단계·상대비4는 사전 고정값. 실패 뒤 범위를 넓히거나 다른 계수를 시도하지 않는다.

## 판정

- 사전 지정 epoch1/3/6/9/13, 각 seed1000/1001/1002의 동일 home 초기 상태 전체 실행.
  최종 지표는 epoch13이다. 중간 최고 checkpoint를 골라 최종 성능으로 쓰지 않는다.
- teacher 인계 없이 전체 student 제어. 기존 품질·접촉획득·유지·완주 기준 그대로.
  조기 종료와 품질 실패도 분모에 포함한다.
- 최종모델에서 기존8관측×3seed의 첫1/8/64행동 오차도 같은 방식으로 평가한다.
  첫8팔오차가 나아져도 전체 완주가 없으면 학습 과제 해결로 표시하지 않는다.
- weighted train_loss와 기존 train_loss를 직접 비교하지 않는다. 비교 진단은 같은
  원래 normalized MSE와 rad 오차를 사용한다.
- 고정 단일 시범에 대한 학습 진단이며 새 물체·배치·언어 일반화의 증거가 아니다.
- 이전 A 실험 자료와 같은 입력·초기화·평가 계약인지 해시로 재확인한다.

## 실행 준비

lecun 코드: `/home/yspark/projects/DexPrompt/out/student-x0-low-noise-code-v1`.
설정: `configs/hx5_student.single-bottle-x0-low-noise.lecun.json`.
학습 데이터: `/home/yspark/projects/DexPrompt/out/student-single-bottle-v1`.
cache: 같은 부모의 `student-single-bottle-v1-cache`.
새 출력 예정: `/data1/yspark/projects/dexprompt/student-runs/single-bottle-x0-low-noise-v1`.
기존 checkpoint·코드·데이터에 덮어쓰지 않는다.

실행 승인 후 lecun GPU1이 비어 있음을 확인하고 tmux에서 시작했다. GPU 번호를 고정해
다른 작업 위에 무조건 올리지 않는다. 학습3시간 상한, 단계별 종료 코드·timeout 기록,
부분 출력은 INCOMPLETE.json으로 그 자리에서 격리하고 학습 실패 시 후속 평가를 중단한다.
평가는 학습 완료 뒤 자동으로 이어지며 그 후 최종 checkpoint의24개 teacher 관측 진단을 한다.

## 남은 작업

1. 완료: 새 코드·설정·데이터 해시 검증과 tmux 학습/평가 래퍼를 연결했다.
2. 학습·고정 평가를 실행하고 최종 epoch13 기준 A/B를 보고한다.
3. 개선 유무에 따라 다음 판단을 한다. 이번 B가 실패하면 추가 계수 탐색이나 HOPE 확대
   학습을 자동 시작하지 않는다.
