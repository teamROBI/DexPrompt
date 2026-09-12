<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# lecun 서버에서 시각 student 평가

2026-09-10. 사용자가 새 서버의 환경 설정과 tmux 실험 실행을 승인했다.
새 학습 없이, 기존 bottle-can BC epoch 2 체크포인트를 평가한다.

- 서버: lecun, `yspark@147.47.200.171:19423`.
- 코드: `/home/yspark/projects/DexPrompt`, 초기 commit `5637da3`, 초기 git status clean.
- 전송 묶음: `/data1/yspark/projects/dexprompt/bottle-can-eval-v1.tar`.
  전송 종료 코드 0, 원격 SHA256 OK.
- GPU: RTX 3090 0번, UUID `GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035`.
  시작 시 GPU 0은 이용률 0%, 약 24GB 여유. GPU 1–5는 다른 작업이 사용 중이므로 사용하지 않는다.
- 드라이버: 580.126.18. Python 3.10.12. EGL/OpenGL/OSMesa 시스템 라이브러리 있음.
- CPU 스레드 1, nice 10, ionice idle. GPU 한 장만 노출한다.

## 설치

최초 venv 생성은 시스템 `ensurepip` 누락으로 실패했다. 원래 실패 로그를 보존하고,
`setup_student_server.sh`에 `venv --without-pip` 및 해당 가상환경 안에서 공식 PyPA
`get-pip.py`를 실행하는 경로를 보완했다. 시스템 Python·드라이버·apt 패키지는 변경하지 않았다.
동일 보완을 기존 서버 코드와 lecun clone에 반영했다. 모델·전처리 소스는 변경하지 않았다.

설치는 `.venv-student`에 수행하고 pip 캐시는 `/data1/yspark/projects/dexprompt/pip-cache`에 둔다.
고정한 torch 2.6.0/cu124, torchvision 0.21.0/cu124, mujoco 3.3.3 등의 설치와
`pip check`가 통과했다. 자산 해제 후 258개 파일 해시·dataset 계약·학습 소스 해시 및
실제 DINO 포함 체크포인트 CPU 로드도 통과했다. 물리 대조군도 4개 모두 통과했다:
기록 행동 4/4 teacher-compliant lift, 제자리 유지 0/4 sustained lift.
근거: [verify.json](verify.json), [controls.json](controls.json),
[설치 버전](requirements.freeze.txt). 설정 종료 코드는 0이다.

## tmux와 중지 조건

- `dexprompt-setup`: 환경 설치 → 자산 해제 → 파일/모델 검증 → 기록 행동 대조군 4개.
- `dexprompt-eval-queue`: 설정 성공을 확인한 뒤 아래 평가를 순서대로 실행.
- `dexprompt-smoke`: 첫 과제, seed 1000 한 번.
- `dexprompt-final`: 고정 과제 4개 × seed 1000–1002, 총 12회.

설정 실패, GPU 사용 충돌, CUDA/EGL 사전 검사 실패 시 다음 단계로 진행하지 않는다.
smoke의 물리적 파지 실패는 결과로 보존하고 본 평가를 진행한다. 알 수 없는 실행 예외는
자동으로 파지 실패에 합산하지 않고 대기열을 중지해 확인한다.
기존 성공 기준·controller·체크포인트·데이터는 바꾸지 않는다.

원격 로그와 상태:

```text
out/server-setup-v1/setup.log, verify.json, exit-code.txt
out/server-controls-v1/summary.json
out/server-evaluation-v1/queue.log, launch.log, exit-code.txt
out/eval3090-smoke-v1/status.json, results/summary.json
out/eval3090-final-v1/status.json, results/summary.json
```

SSH 단절 시 tmux를 재시작하지 말고 위 상태를 먼저 확인한다.
21:09:23 KST smoke 시작, 3090 GPU 0의 CUDA 계산·EGL 렌더 사전 검사 통과.
21:11:18 KST smoke는 종료 코드 0으로 완료됐다. 결과는 0/1 완주:
seed 1000, 17 tick, `penetration`, 최대 물체 침투 6.54mm, 최대 손가락 힘 75.5N.
원시 기록: [smoke.json](smoke.json). 환경 실행 실패가 아닌 물리적 파지 실패다.

대기열에 추가한 검사에서 `quality_` 접두사가 붙는다고 잘못 가정해 본 평가가 멈췄다.
실제 `quality_violation()` 반환값 6종(`penetration` 등)에 맞게 분류를 수정했다.
최초 대기열 로그·exit-code를 `*-classifier-error.*`로 보존하고, `--after-smoke`로
기존 smoke 결과를 검사한 뒤 본 평가만 이어서 시작했다. 모델·평가 기준은 변경하지 않았다.
본 평가는 21:14:13 KST `dexprompt-final` tmux에서 시작됐고, CUDA/EGL 검사도 통과했다.
21:21:12 KST 정상 종료(exit 0), **기록 12회에서 sustained lift 0, teacher-compliant lift 0**.
병 9개 기록은 모두 penetration(각 seed의 tick 17/13/16), 캔은 penetration 100 tick,
table_collision 118/77 tick에서 종료했다. 대기열도 종료 코드 0이다.
원시 집계: [final.json](final.json), 실행 상태: [status.json](status.json).

분모 정정: 병 3시범의 초기 RGB+mask, wrist RGB, measured joints, mask·가시성 배열이
전부 동일하다([초기 입력 감사](initial-input-audit.json)). 따라서 "4개 시범 기록 × 3회"로
보고하며 4개 서로 다른 정책 입력 과제나 12개 독립 조건으로 주장하지 않는다.
전체 reset·목표 조건·판정 계약의 동등성과 궤적 중복은 다음 진단에서 확인한다.
이 결과는 train-task 진단이며 새로운 물체·배치 일반화와 구분한다.

다음은 재학습 전에 초기 명령·전처리·동작 chunk·정규화와 중복 평가 계약을 감사하는 것이다.
실행 계획과 Claude 분담은 [다음 작업 문서](../../../plans/NEXT_WORK_2026-09-10.md)에 있다.
