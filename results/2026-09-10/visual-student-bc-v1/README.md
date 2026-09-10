# 병·캔 시각 student BC — 공용 서버 실행

2026-09-10. 사용자가 학습 시작과 tmux 실행을 승인했다.
13:45 KST에 `icra-student-bc` 세션을 시작했고, 데이터 검사 후 첫 실제
optimizer 업데이트와 유한한 손실을 확인했다. 학습은 15:47:09 KST에 종료 코드 0으로 완료됐고 완주 성능은 미평가다.

## 고정 조건

- 설정: `configs/hx5_student.bottle-can.local.json`.
- 데이터: `out/student-bottle-can-v1`, cache: `out/student-bottle-can-v1-cache`.
- 병 3 + 캔 1 시범, 4,170프레임. batch 2, 3 epoch, 총 6,255 update, seed 42.
- 학습 과제 재현용 진단이며 val/test 없음. 새 물체 일반화 결과로 쓰지 않는다.
- teacher의 닫기 제어에 인계하는 팔 전용 정책이 아니라 시각 입력의 27관절 student다.
- 최종 평가 체크포인트는 사전에 정한 `epoch_0002.pt`. 평가 자동 실행은 연결하지 않았다.

## 자원 사용

- 물리 GPU 0 한 장만 UUID `GPU-4e217f74-0659-cfbc-d0fb-41b165be97c5`로 노출.
- PyTorch CUDA allocator 메모리 상한 0.5(약 24GB). CUDA 외부 할당까지 포함한
  GPU 전체 점유에 대한 엄격한 상한은 아니다.
- CPU/BLAS 스레드 1, DataLoader worker 0, nice 10, ionice idle(3).
- 각 학습 스텝 계산 시간의 1배를 휴지 시간으로 추가. 명목상 계산/휴지 1:1이며
  다른 프로세스에 GPU 시간 50%를 보장하는 스케줄러 설정은 아니다.
- 다른 사용자의 프로세스는 변경하지 않았다.
- 홈 디스크 여유 약 6.2GB로 체크포인트 3개 저장이 부족해, 당시 약 333GB가
  남은 `/data1/yspark/sim/icra/student-runs/bottle-can-bc-v1`에 출력한다.

## 재접속과 진행 확인

```bash
tmux attach -t icra-student-bc
```

화면만 빠져나오기: `Ctrl-b`를 누른 뒤 `d`. 세션에서 `Ctrl-c`는 학습을 중단한다.

```bash
tail -f /home/yspark/projects/ICRA/out/student-bottle-can-launch-v1/train.log
cat /data1/yspark/sim/icra/student-runs/bottle-can-bc-v1/progress.json
```

- 실행 스크립트: `out/student-bottle-can-launch-v1/launch.sh`.
- `progress.json`: 10 update마다 손실·진행·경과 시간·잔여 시간 추정, 원자적으로 교체.
- `run.json`: 실제 첫 optimizer.step 뒤 `training_started: true`와 UTC 시각 기록.
- `metrics.tsv`: epoch별 평균 손실. epoch 종료 전에는 아직 비어 있을 수 있다.
- `epoch_0000.pt`~`epoch_0002.pt`: 모델·optimizer·scheduler 포함, 임시 파일을
  완성한 뒤 원자적으로 이름을 바꾼다.
- 학습 프로세스 종료 뒤 `out/student-bottle-can-launch-v1/exit-code.txt` 생성.
  0은 정상 종료, 다른 값은 로그 확인. 종료 시 tmux 세션도 사라질 수 있다.

네트워크 단절에는 tmux로 계속 실행된다. 서버 재부팅·프로세스 종료까지 견디는
자동 복구 기능은 없다. 현 trainer에는 checkpoint resume 옵션이 없으므로
중단 시 산출물을 보존하고 복구 경로를 별도로 준비해야 한다.

## 코드 보완

`train_hx5_student.py`에 자원 제한 옵션, 10-step 진행 기록, 실제 training_started
기록, JSON 및 epoch 체크포인트 원자적 저장을 추가했다. 모델·학습률·손실은
바꾸지 않았다. 기본 rest ratio는 0, 메모리 비율은 1이라 기존 실행은 제한이 없다.
관련 테스트 43개 통과, Ruff 및 diff whitespace 검사 통과 후 실제 실행했다.
학습 중인 source 파일은 변경하지 않는다. checkpoint가 기록한 해시와 평가 코드의
일치 검사를 유지한다.

## 다음 작업

학습 종료 코드 0 및 세 체크포인트 확인 후 기존 `evaluate_student_suite.py`로
4개 시범 × 3회 폐루프 평가한다. 출력 경로는 이번 `/data1` 체크포인트에 맞춘다.
손실 감소와 파지 완주를 구분하고, 실패 단계·접촉 유지·목표 가림을 함께 확인한다.

## 시작 후 점검

100 update 시점: 누적 평균 손실 0.6238, 휴지 포함 0.821초/update, 잔여 약 85분.
GPU 0 총 점유 11,222MiB(시작 전 4,767MiB), GPU 1–3 메모리 점유는 동일했다.
순간 GPU 0 사용률 26%, 프로세스 CPU 평균 67%(한 코어=100%)였다.
다른 작업의 실제 처리량을 측정한 것은 아니므로 영향이 0이라고 보장하지 않는다.
초기 추정이며 checkpoint I/O와 공용 서버 부하에 따라 완료 시각은 달라진다.

## 학습 완료

2026-09-10 15:47:09 KST 정상 종료(약 2시간 2분). 6,255 optimizer updates,
3 epoch. 평균 학습 손실은 0.105492766 → 0.0285537934 → 0.0171776801.
약 3.4GB씩 epoch 체크포인트 3개 존재, run.json의 모든 학습 소스 해시가
현재 파일과 일치함을 확인했다. loss 감소는 파지 성공 증거가 아니다.
다음은 epoch_0002.pt의 폐루프 평가이며 아직 실행하지 않았다.
