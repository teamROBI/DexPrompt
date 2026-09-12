<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# lecun 단일 병 시범 재현 학습

사용자가 다음 작업 실행을 지시했다. 기존 서버는 사용하지 않고 lecun GPU 0에서
데이터 준비→사전 검증→대조군→제한된 학습→사전 지정 평가를 tmux로 실행한다.
학습/평가 완료 결과를 아래에 기록한다.

## 완료 결과 — 09-11 확인

원격 `status.json`은 `phase: completed`, `running: false`다. 학습 6,331회 갱신과
예정된 epoch 1/3/6/9/13 평가가 모두 끝났다. 각 checkpoint를 같은 초기 과제에서
추론 seed 1000–1002로 평가했으며 **전부 0/3 완주**다. 이는 새로운 물체의 일반화
평가가 아니라 학습한 단일 과제의 재현 검사다.

사전 지정 최종 epoch 13에서도 완주 0/3, 처음 명령의 teacher 대비 최대 관절 오차는
0.414/0.918/0.860 rad다. 각각 213/18/24 tick에 digit_force/penetration/
approach_displacement로 끝났다. 마지막 progress 손실은 0.007818로 감소했지만
폐루프 전이가 성공했다는 증거는 없다. 추가 학습은 시작하지 않았다.

원시 집계는 `out/single-bottle-job-v1-server/results.json`, 작은 요약은
[completed-summary.json](completed-summary.json)에 보존했다. 아래 실행 시작 기록은
과거 시점 기록이다. HOPE 자산 준비와 별개로 이 단일 시범 전이 문제를 해결해야 한다.

## 사전 고정 설정

- 선택: 기존 `source0_teacher_0002_yaw_000`, 974 frame, 병 1개·시범 1개.
- 원본: `out/student-bottle-can-v1`, 원본 시범과 기존 checkpoint를 보존한다.
- 새 데이터: `out/student-single-bottle-v1`; 캐시: `out/student-single-bottle-v1-cache`.
- `prepare_student_diagnostic.py --episode-id`로 명시적으로 선택한다.
  잘못된 ID나 여러 입력에서의 모호한 선택은 출력 생성 전에 실패한다.
  실제 시범의 모든 NPZ 배열과 품질 판정을 원본과 대조한 뒤 캐시를 만든다.
- 설정: `configs/hx5_student.single-bottle.lecun.json`.
  기존 모델 구조·카메라·목표 기억·diffusion 설정, LR 1e-4, batch 2, seed 42 유지.
  고정 DINO 가중치를 사용하고 diffusion은 새로 초기화한다.
- 상한: 13 epoch / 6,331 optimizer updates, 학습 subprocess wall time 3시간.
  데이터 축소로 normalizer도 바뀌므로 엄밀한 단일 요인 인과 비교라고 하지 않는다.
- 학습 출력: `/data1/yspark/projects/dexprompt/student-runs/single-bottle-v1`.
- GPU: `GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035`(lecun 0번).
  시작 전 15MiB/0%, `/data1` 약 584GB 여유. 다른 GPU는 사용 중이라 건드리지 않았다.
  준비와 학습 직전에 GPU 점유를 확인한다. CPU/BLAS 1스레드, nice 15.
- 검증: 신규 선택/래퍼를 포함한 관련 테스트 12개, 변경 코드 Ruff 통과.
  원래 모델·학습기·실행기 소스와 품질 기준은 수정하지 않았다.

## 실행과 중지

tmux `dexprompt-single-bc`, 루트 `/home/yspark/projects/DexPrompt`.
실행기: `scripts/run_single_student_lecun.py`.

진행 상태:

```text
out/single-bottle-job-v1/status.json
out/single-bottle-job-v1/selection.json
out/single-bottle-job-v1/<stage>.started.json
out/single-bottle-job-v1/<stage>.finished.json
out/single-bottle-job-v1/<stage>.log
out/single-bottle-launch-v1.log
/data1/yspark/projects/dexprompt/student-runs/single-bottle-v1/progress.json
/data1/yspark/projects/dexprompt/student-runs/single-bottle-v1/run.json
```

단계별 exit code와 timeout을 기록한다. 실패하면 다음 단계로 넘어가지 않으며,
timeout은 해당 subprocess group만 중지한다. 부분 checkpoint는 평가하지 않는다.
현재 래퍼에서 학습 시간 상한에 걸리면 자동 평가까지 중지하고 완성된 checkpoint를
보존한다. 이 경우 사전 지정 최종 epoch 13 결과를 다른 epoch로 대체하지 않는다.
네트워크만 끊기면 tmux 작업은 계속되므로 새 작업을 중복 실행하지 않는다.

## 학습 후 평가

epoch **1/3/6/9/13**, 각 seed **1000/1001/1002**, 같은 초기 상태의 폐루프 평가다.
epoch 13이 사전 지정 최종 결과이며 중간 epoch는 진단용이다.
각 checkpoint 평가의 시간 상한은 1시간으로 학습 3시간 예산과 별도다.
최악의 총 상한은 준비 단계와 평가를 포함하면 3시간보다 길다.

품질 판정과 첫 명령의 시범 오차를 `out/single-bottle-job-v1/results.json`에 남긴다.
알 수 없는 실행 예외를 파지 실패로 합산하지 않는다. 완료 조건은 teacher 제어기로
인계하지 않고 카메라+목표 mask+관절만으로 home→접근→닫기→지속 lift를
기존 접촉/품질 기준으로 통과하는 것이다. 학습 시범 자체의 재현 진단이며 일반화 평가가 아니다.

## 학습 시작 확인

준비·캐시·preflight·대조군이 모두 통과해 실제 optimizer 갱신이 시작됐다.
`training_started: true`, 100 updates에서 평균 train loss 0.5829로 유한하다.
초기 평균 0.545초/update이며 이 속도에서 남은 학습은 약 57분이다.
checkpoint 저장과 평가 시간은 별도이며 완료 시각을 보장하는 추정은 아니다.

- 선택한 974프레임의 **모든 NPZ 배열이 원본과 동일**하다.
- preflight: 누락 항목 없음, 참조 미확보 prefix 0, train 1/val 0/test 0.
- 기록 행동 대조군: 974 tick, 14.64cm 상승, 종료 접촉 2/2,
  접촉을 유지한 마지막 lift 2.76초. 제자리 대조군은 lift 없음.
- 새 manifest SHA: `24233706f31ca4f241a8135e69348f2f7fe8344575946f9daec6cdb87b64f135`.
- cache manifest SHA: `bad6aba5e5b243200428bdae2941f58ff10286f040cff5172bc237f86051e62d`.

현재 상태는 원격 status/progress 파일로 확인한다. 위 수치는 학습 시작 점검 당시의 기록이다.
