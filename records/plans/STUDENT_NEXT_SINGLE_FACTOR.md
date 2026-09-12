<!-- research-scope-2026-09-12 -->
> **날짜별 실험 계획 기록:** 현재 목표·우선순위는 [연구 정의](../../RESEARCH_PLAN.md)와
> [현재 작업서](../../plans/CURRENT_WORK.md)가 우선한다. 아래 실행 지시·P번호·미실행/진행 상태는 작성 당시 것이다.
> 고정 후보 접근 PPO는 형상 기반 선정/적응 RL 전체와 다르다. 부위 지시는 필수 범위가 아니다.
> 완료된 작업을 이 문서만 보고 재시작하지 않는다. 기존 승인 및 원본 보존은 유지한다.

# 다음 단일 요인 실험 — teacher 범위의 관절별 명령 변화 제한

작성일 2026-09-11. 이 문서는 실행 계약이다. **실험 완료: 완주 0/3.** 결과는 [student-slew-v1](../experiments/2026-09-11/student-slew-v1/README.md)에 있다. 아래는 고정 실행 계약이며 새 학습,
optimizer 갱신, 모델 선택, HOPE 데이터 사용은 포함하지 않는다.

## 질문

단일 병 x0 최종 정책의 실패는, 예측한 절대 관절 명령이 teacher보다 한 tick에 너무 크게
변한다. 고정 변화량 제한이 초기 이탈과 최종 완주를 개선하는가?

기존 평가만으로는 답할 수 없다. 세 실행 모두 최초 입력이 teacher tick 0과 bit-identical인데
첫 명령부터 arm joint 1이 0.158–0.170 rad 틀렸다. 첫 64 tick의 재계획 경계 변화량
중앙값은 구간 내부의 26.6–33.3배였고, teacher에서는 1.00배였다. 그러나 기존 실행은
잘못 예측한 명령의 크기와 방향을 모두 그대로 적용했으므로, 큰 변화량 자체가 실패에
필수였는지 분리하지 못했다.

## 바꾸는 한 요인

**Student가 낸 절대 명령을 simulator에 적용하기 직전, 관절별 한 tick 변화량만 제한한다.**
이전 적용 명령을 `u(t-1)`, 모델 출력을 `u_raw(t)`, 고정 한계를 `d`라 하면

```text
u(t) = clip(u_raw(t), u(t-1) - d, u(t-1) + d)
```

첫 tick의 `u(t-1)`은 측정 home 관절 상태다. `d`는 같은 단일 병 teacher 시범에서 관절별
`max(|action[0]-state[0]|, |action[t]-action[t-1]|)`로 한 번 계산하고 고정했다.
온라인 teacher 상태·단계·접촉·목표 자세는 사용하지 않는다. 고정 파일은
[`STUDENT_NEXT_SLEW_LIMITS.json`](../../plans/STUDENT_NEXT_SLEW_LIMITS.json), SHA256은
`e779ea9740107247b3ec6d101b93efc1daf22521ed6069cd89f3c9a8133f6a7d`다. 한계 범위는
1.62e-6–0.04254 rad/tick이며 source teacher의 974개 명령은 하나도 바꾸지 않는다.

다음 항목은 바꾸지 않는다.

- x0 epoch 13 체크포인트와 SHA256
  `a5100b38f9c1fb6440386a70da4acf6023c1a91f8d9ff10d4c6d932686244aca`.
- 단일 병 dataset manifest SHA256
  `24233706f31ca4f241a8135e69348f2f7fe8344575946f9daec6cdb87b64f135`.
- 최초 simulator state/model, 카메라, DINO, mask/goal memory, action horizon 64,
  execute steps 8, diffusion inference steps 16.
- seed 1000/1001/1002, 각 974 tick, 동일 contact-acquisition guard와 모든 품질 한계.
- 최종 epoch 고정. 결과를 보고 다른 checkpoint·한계·seed를 고르지 않는다.

## 준비된 구현

[`replay_hx5_student.py`](../../scripts/replay_hx5_student.py)는 기본값에서 기존 명령을 그대로
사용한다. `--slew-limits`를 명시한 새 진단 실행에서만 postprocessor를 적용한다. 모델과
전처리 소스는 바꾸지 않았으므로 checkpoint의 source 검사를 우회하지 않는다.

- 제안 명령은 실제 recorder로 보내기 전에 가로채고, 제한된 명령만 한 번 보낸다.
- trace에 제한된 action과 pre-action state를 같이 남긴다.
- actuator 범위 clipping과 slew clipping 횟수를 별도로 기록한다.
- wrapper·teacher 한계 산출 관련 회귀 테스트를 통과했다.
- evaluator script SHA256:
  `145de613d45a1068a1b20ae7d5f7bf2638c54e718226e5e2c9ce795aee186aab`.

## 대조군과 판정

1. **양성 대조군:** 한계 산출기가 원본 teacher의 모든 명령을 byte-level 값으로 보존해야
   한다. 현재 `teacher_commands_changed_by_limit=0`이다. 기존 기록 명령은 같은 simulator에서
   완주한다. 어느 하나라도 달라지면 student 결과를 해석하지 않는다.
2. **음성 기준선:** 같은 x0 epoch 13, 같은 세 seed의 기존 결과는 완주 0/3,
   최대 동시 접촉군 0/1/1, 모든 필요 접촉군 동시 tick 0, 최대 상승 1 mm 미만이다.
3. **시험군:** 고정 limiter를 적용한 같은 세 seed만 한 번 실행한다.

주 지표는 `teacher_compliant_lift` 완주 수/3이다. 보조 지표는 contact acquisition tick,
최대/종료 접촉군, 10 cm 이상 연속 상승 시간, 품질 위반, `slew_limited_ticks`, 최초 64 tick
arm/hand 상태 오차, 재계획 경계와 내부의 적용 명령 변화량이다.

- 0/3이고 접촉도 기준선 수준이면 “과도한 변화량만으로는 실패를 설명하지 못한다.”
- 접촉 획득은 늘지만 완주 0/3이면 접근 이탈에는 기여하지만 닫기·운반 실패가 남는다.
- 1/3 이상 teacher-compliant 완주면 큰 명령 변화가 이 고정 과제 실패의 인과적 요인이었다는
  진단 근거가 된다. 한 물체·한 체크포인트이므로 일반화나 seed 강건성을 주장하지 않는다.

어떤 결과에서도 한계를 다시 조정하거나 다른 checkpoint를 이어서 시험하지 않는다. 다음
분기는 새 실험으로 다시 고정한다.

## lecun 실행 절차

기존 `student-x0-code-v1`의 파일을 덮어쓰지 않는다. 새 evaluator 이름과 새 결과 경로를 쓴다.
먼저 파일을 `.partial`로 전송하고 SHA를 확인한 뒤 이름을 바꾼다.

```bash
# 로컬에서, 실행 직전 GPU UUID와 출력 경로가 비었는지 다시 확인한다.
scp -i out/server-transfer/lecun/id_ed25519 -P 19423 \
  scripts/replay_hx5_student.py \
  yspark@147.47.200.171:/home/yspark/projects/DexPrompt/out/student-x0-code-v1/scripts/replay_hx5_student_slew.py.partial
scp -i out/server-transfer/lecun/id_ed25519 -P 19423 \
  plans/STUDENT_NEXT_SLEW_LIMITS.json \
  yspark@147.47.200.171:/data1/yspark/projects/dexprompt/student-checks/STUDENT_NEXT_SLEW_LIMITS.json.partial
```

원격에서 두 SHA가 위의 고정값과 같은지 확인한 다음 `.partial`을 최종 이름으로 바꾸고,
아래를 tmux에서 실행한다.

```bash
tmux new-session -d -s dexprompt-x0-slew-v1 \
  'cd /home/yspark/projects/DexPrompt/out/student-x0-code-v1 && \
   exec env CUDA_VISIBLE_DEVICES=GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035 \
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
   /home/yspark/projects/DexPrompt/.venv-student/bin/python \
   scripts/replay_hx5_student_slew.py \
   --data /home/yspark/projects/DexPrompt/out/student-single-bottle-v1 \
   --episode source0_source0_teacher_0002_yaw_000 \
   --checkpoint /data1/yspark/projects/dexprompt/student-runs/single-bottle-x0-v1/epoch_0012.pt \
   --device cuda:0 --seed 1000 --repeats 3 --allow-training-overlap \
   --slew-limits /data1/yspark/projects/dexprompt/student-checks/STUDENT_NEXT_SLEW_LIMITS.json \
   --out /data1/yspark/projects/dexprompt/student-checks/x0-slew-v1'
```

기존 같은 3회 평가는 805.7초였다. 시험군도 약 14분으로 예상하되 tmux wall-time 상한은
1시간으로 둔다. 학습 예산은 0 step이고 backward/optimizer는 호출하지 않는다. 실행 후에는
새 trace를 G1과 같은 분석으로 비교하고, 해시·세 seed·최대 974 tick 예산·고정 한계를 확인한다. 품질 위반에 의한 조기 종료는
실패로 분모에 포함한다. 통제 설정의 불일치는 별도 오류로 처리한다.
