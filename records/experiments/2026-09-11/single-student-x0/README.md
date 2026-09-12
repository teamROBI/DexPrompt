<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 단일 병 시각 student: epsilon → x0 비교

2026-09-11. 사용자가 실행을 지시했고, lecun GPU가 사용 중임을 알린 뒤
"아니야 lecun 그냥 먹자"로 GPU 공유 실행을 명시했다.

## 질문과 고정 조건

기존 단일 병 epsilon 정책은 6,331회 갱신 후 최종 완주 0/3이었다.
국소 teacher-noised 입력의 작은 오차와 순수 잡음에서의 큰 행동 생성 오차가
함께 관측됐으므로, 이번에는 정규화된 clean action(x0)을 직접 예측한다.
이 가설이 맞는지 아직 결과는 없다.

- 같은 단일 병 시범 974 tick, 같은 dataset/cache 및 normalizer.
- 같은 실제 DINO 두 개, 모델 폭/깊이, action horizon 64/execute 8.
- 같은 cosine 50단계, DDIM 16단계, [-1,1] clean estimate clipping.
- 같은 seed 42, batch 2, 13 epochs / 6,331 optimizer updates, 학습률과 optimizer.
- epoch 1·3·6·9·13 평가, 각각 seed 1000·1001·1002. 최종 지표는 사전 지정 epoch 13.
- teacher 인계 없이 기존 simulator 전체 실행 및 동일 quality/접촉/성공 판정.
- 데이터 추가, 관측/목표 기억/제어 변경, 기존 체크포인트 미세조정 없음.

`configs/hx5_student.single-bottle-x0.lecun.json`은 기존 설정에
`prediction_type: x0` 하나만 추가한다. 학습 손실은 x0 MSE로 바뀌고 sampler는
그 출력에서 epsilon을 계산한다. 원시 x0로 epsilon을 계산한 뒤 clean estimate만
기존과 같이 clipping한다. **균등 x0 MSE는 epsilon MSE와 시간 단계별 가중 효과도
다르므로 두 실험의 loss 숫자를 직접 비교하지 않는다.** 판정은 전체 파지 실행이다.

기존 설정에서 키를 생략하면 epsilon이며, 기본값과 명시 epsilon의 loss/출력은
동일하다. 새 run/config를 체크포인트에 저장하고 strict load에서 x0를 복원한다.
기존 epsilon 체크포인트를 x0로 재해석하지 않는다.

## 구현 검증

관련 테스트 **82개 통과**, 그중 신규 16개. 변경 코드 Ruff 및 diff check 통과.
- x0 정답 oracle: 1/16/50 inference steps에서 최종 절대 행동과 중간 DDIM 상태 확인.
- normalized x0 loss, padding 입력 불변성, padding gradient 0 확인.
- epsilon 기본값과 명시 설정 동일, 기존 epsilon oracle 유지.
- x0 체크포인트 strict load 및 동일 출력 확인.
- 비교 설정 한 요인 검사, 기준선 소스 변경 거부, 공유 GPU 메모리/STOP/대기 상한 검사.
작은 합성 텐서의 gradient 검사는 로컬에서 수행했으며 실제 모델 학습은 lecun 전용이다.

## 실행 위치 및 자원

- tmux: `dexprompt-x0-bc`
- 코드 사본: `/home/yspark/projects/DexPrompt/out/student-x0-code-v1`
- 진행/로그: `/home/yspark/projects/DexPrompt/out/single-bottle-x0-job-v1`
- 모델: `/data1/yspark/projects/dexprompt/student-runs/single-bottle-x0-v1`
- GPU 0: `GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035`.
- 공유 실행, 학습과 전체 모델 dry-run의 PyTorch 메모리 상한 85%(약 20.4 GiB).
  시작 시 free memory 22,000 MiB 이상 확인. 다른 프로세스는 수정/종료하지 않는다.
- BLAS/OMP 1 thread, nice 15, 학습 3시간 상한. 메모리 부족 시 각 GPU 단계 앞에서
  최대 8시간 기다린 뒤 실패 종료한다. GPU 연산 자원을 독점하거나 예약하는 장치는 아니다.
- 사전 검사·기록 행동/정지 대조군 → 실제 DINO full-model dry-run → train → 자동 평가.
  각 단계 종료 코드/timeout 기록, 실패하면 다음 단계로 넘어가지 않는다.

코드 사본은 lecun 기존 코드를 복사한 뒤 모델 파일과 새 config/runner만 적용했다.
기존 epsilon run이 해시로 기록한 소스 중 변경된 것은 `src/icra/student.py`뿐이다.
원래 checkout과 체크포인트는 그대로라 과거 결과를 strict load로 재현할 수 있다.
HOPE 생성·클로드 작업 경로는 바꾸지 않았다.

## 현재 상태

**실제 학습 시작 확인: 60/6,331 optimizer updates, 유한 train loss 0.19183.**
사전 검사와 기록 행동/정지 대조군 통과. 실제 DINO full-model dry-run도 유한 손실과
`[2,64,27]` 출력을 확인했고 backward/optimizer step 0회, 모델 불변이었다.
새 run.json의 `training_started: true`를 확인했다. 기준선 대비 dataset/cache,
normalizer, DINO weights, joint ordering, control dt가 모두 동일하다.
확인 시 GPU 0 총 메모리 7,801 MiB(기존 사용자 작업 포함), 현재 속도상 학습 약 2시간.
이후 자동 평가 시간이 추가된다. 실행 결과는 아직 대기 중이다.

`startup.json`은 이 시점의 기록이며 실시간 상태 파일이 아니다.
`epsilon-preservation.json`: lecun 원본 모델과 새 코드 기본 epsilon 경로를 작은
CPU fixture로 직접 비교해 초기 파라미터·loss·행동 출력 모두 bit 동일함을 확인했다.
`status.json`, `train.started.json`, 모델 디렉터리의 `run.json`/`progress.json`을
구분한다. tmux 실행 자체를 학습 시작이나 학습 성공으로 세지 않는다.
결과가 나와도 단일 학습 과제 진단이며 다물체/새 배치/언어 일반화의 증거는 아니다.
