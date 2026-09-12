<!-- research-scope-2026-09-12 -->
> **날짜별 실험 계획 기록:** 현재 목표·우선순위는 [연구 정의](../../RESEARCH_PLAN.md)와
> [현재 작업서](../../plans/CURRENT_WORK.md)가 우선한다. 아래 실행 지시·P번호·미실행/진행 상태는 작성 당시 것이다.
> 고정 후보 접근 PPO는 형상 기반 선정/적응 RL 전체와 다르다. 부위 지시는 필수 범위가 아니다.
> 완료된 작업을 이 문서만 보고 재시작하지 않는다. 기존 승인 및 원본 보존은 유지한다.

# 단일 병 시범의 시각 재현 진단 — 준비 설정

실행 갱신: 사용자 지시로 `dexprompt-single-bc` tmux의 준비→검증→학습→평가 파이프라인을
시작했다. 최신 상태는 [실행 기록](../experiments/2026-09-10/single-bottle-student/README.md)과
원격 `out/single-bottle-job-v1/status.json`을 참조한다. 아래는 실행 전 고정한 계획이다.

실행 서버는 사용자 지시에 따라 **lecun만 사용**한다. 기존 서버에서 학습하지 않는다.
학습/평가는 SSH 단절에 영향받지 않도록 tmux에서 수행한다.
실제 optimizer 갱신 시작은 학습 출력 `run.json`의 `training_started`로 확인한다.

- 원본 데이터: `out/student-bottle-can-v1`.
- 선택 시범: `source0_teacher_0002_yaw_000`, 974 tick.
  이전 감사에서 기준으로 사용한 첫 병 시범을 고정하며 결과를 보고 바꾸지 않는다.
- 새 데이터/캐시: `out/student-single-bottle-v1`, `out/student-single-bottle-v1-cache`.
  원본을 보존하면서 선택 시범 하나와 archived simulator/소스만 별도 묶음으로 만든다.
- 모델: 기존 실제 DINO+diffusion architecture, 27관절, horizon 64, execute 8,
  inference 16, 같은 camera/goal contract. diffusion은 기존 혼합 데이터 checkpoint를
  이어 쓰지 않고 seed 42로 새로 초기화한다. DINO는 기존 고정 가중치를 사용한다.
- 설정: `configs/hx5_student.single-bottle.lecun.json`.
  batch 2, LR 1e-4, warmup 100, **13 epoch / 6,331 updates 상한**.
  기존 4시범 학습의 6,255 updates와 비슷한 계산 예산의 재현 진단이다.
  13×ceil(974/2)=6,331이며, 데이터 준비 뒤 실제 dataset 길이로 다시 확인한다.
  데이터가 바뀌면 normalizer도 재계산되므로 단순한 한 요인 인과효과 비교라고 하지 않는다.
- GPU는 lecun에서 비어 있는 장치만 사용한다. 기존에 사용한 GPU 0의 UUID는
  `GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035`지만 실행 직전에 다시 확인한다.
  다른 사용자 프로세스는 중지/이동하지 않는다. CPU/BLAS 1스레드, nice 15.
- 출력은 lecun `/data1/yspark/projects/dexprompt/student-runs/single-bottle-v1`.
  기존 trainer는 epoch마다 optimizer를 포함해 저장하므로 약 45GB가 필요하다.
  데이터/캐시/임시 checkpoint를 포함해 시작 전 여유 60GB 이상을 확인한다.

실행 전에 할 일:

1. 단일 시범 묶음과 cache를 만들고, 시범 SHA/관절/품질/시뮬레이터 계약을 검증한다.
2. lecun에서 preflight와 기록 명령 양성 대조군을 실행한다.
3. 최대 wall time 3시간, 비유한 손실/실행 예외 시 중지하는 tmux 실행 래퍼를 준비한다.
   시간 상한은 실제 성능 예측이 아니다. 실패를 성공으로 보이게 하는 gate 변경은 하지 않는다.
4. 학습 후 epoch 1/3/6/9/13의 고정 초기 관측 명령 오차와 seed 1000–1002의
   폐루프 결과를 비교한다. 최종 epoch는 사전 지정 결과, 중간 epoch는 진단으로 구분한다.
   모두 같은 초기 상태의 잡음 반복이며 독립 물체나 일반화 평가가 아니다.
5. 완료 조건은 teacher 실행기로 인계하지 않고 home→접근→닫기→지속 lift를
   기존 요구 접촉·침투·힘 기준으로 통과하는 것이다. loss 감소만으로 완료하지 않는다.

위 데이터/캐시와 실행 래퍼를 준비하기 전에는 이 config만으로 학습을 실행하지 않는다.
Claude의 그릇 데이터는 이 진단에 추가하지 않는다.
