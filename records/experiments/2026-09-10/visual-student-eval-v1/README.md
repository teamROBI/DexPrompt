<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 병·캔 시각 student 첫 폐루프 평가

2026-09-10. 사용자 요청으로 16:18 KST `icra-student-eval` tmux에서 시작했다.
GPU 4장이 다른 사용자의 학습으로 약 99% 사용 중이므로 CPU 평가를 선택했다.
이후 사용자 요청으로 CPU 평가를 중지했다. 최종 결과는 없으며 자동 재시작하지 않는다. 새 학습이나 정책 수정은 하지 않는다.

## 사전에 고정한 조건

- checkpoint: `/data1/yspark/sim/icra/student-runs/bottle-can-bc-v1/epoch_0002.pt`.
  3 epoch/6,255 updates의 최종 checkpoint이며 평가 결과로 고른 것이 아니다.
- dataset: `out/student-bottle-can-v1` (병 3시범, 캔 1시범).
- 각 시범 seed 1000, 1001, 1002, 총 12회. 동일한 archived 초기 상태에서
  diffusion sampling seed만 다르다. 물체·초기 배치의 무작위화가 아니다.
- train-task 진단을 명시한다. 학습 물체 일반화나 실제 로봇 성공률로 보고하지 않는다.
- 원래 시범의 전체 tick 예산, contact_acquisition_guard_v2, 기존 접촉/품질 기준 유지.
- actor는 카메라+mask+측정 관절을 받아 27관절을 직접 명령한다.
  teacher 제어기 인계나 성공을 위한 접촉 보정은 없다.
- 모델·전처리·DINO 가중치 해시 검사를 유지한다. allow-code-change를 쓰지 않는다.
- CPU thread 1, DataLoader worker 없음, nice 10, ionice idle, CUDA_VISIBLE_DEVICES 빈 값.
- MUJOCO_GL=osmesa, 소프트웨어 렌더러 thread 1. 학습 및 기존 렌더 기록과 backend
  차이가 있으므로 결과는 CPU/OSMesa 조건으로 명시한다. GPU seed와 bit 동일을 주장하지 않는다.
- 초기 기록 명령 양성 4/4, 제자리 음성 0/4 검증은
  `../student-training-ready/controls.json`의 같은 평가 계약을 참조한다.

## 실행·재접속

```bash
tmux attach -t icra-student-eval
tail -f out/student-bottle-can-eval-launch-v1/evaluate.log
```

출력: `/data1/yspark/sim/icra/student-runs/bottle-can-bc-v1-eval-cpu`.
시범별 로그·`evaluation.json`·반복별 trace NPZ, 전체 `summary.json`을 확인한다.
시범별 evaluation.json은 해당 시범의 반복 3회를 마친 뒤 작성된다.
`out/student-bottle-can-eval-launch-v1/exit-code.txt`는 프로세스 종료 시 생성된다.
네트워크 단절 후 중복 실행하지 않는다. tmux detach는 Ctrl-b, d.

모든 시범이 끝나면 완주·필수 접촉 준수·실패 단계·상승·침투를 보고한다.
프로세스 오류나 모델 로딩 오류는 파지 실패율에 합산하지 않는다.

## 사용자 요청에 따른 중지

CPU 평가가 오래 걸릴 수 있어 사용자가 나중에 하자고 요청했다.
`icra-student-eval`에 Ctrl-C를 전달했다. 중간 파일은 보존했고, 완료된
학습이나 다른 사용자 프로세스에는 중지 신호를 보내지 않았다.
평가 재개는 새 출력 폴더에서 수행하며 CPU 부분 기록과 GPU 평가를 혼합하지 않는다.
