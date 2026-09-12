<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 낮은 잡음 x0 학습 — 실제 갱신 시작 확인

사용자가 “새 학습 시작하자”로 승인했다. lecun GPU1 UUID432d6dc7…가 비어 있음을 확인하고
22:29에 tmux `dexprompt-low-noise-train-v1`을 시작했다.

- 사전 고정한 low_noise_loss 옵션만 변경. seed42,974프레임,13epochs/6,331updates.
- preflight→기록 행동·정지대조군→실제DINO dry-run→학습→고정epoch1/3/6/9/13 각3seed 평가.
- 최종 지표epoch13. 이후8관측×3seed 행동 오차도 자동 진단한다.
- code `/home/yspark/projects/DexPrompt/out/student-x0-low-noise-code-v1`.
- job `/data1/yspark/projects/dexprompt/student-checks/single-bottle-x0-low-noise-job-v1`.
- 모델 `/data1/yspark/projects/dexprompt/student-runs/single-bottle-x0-low-noise-v1`.
- `status.json`은 단계 상태, `*.started.json`은PID·명령·시간 상한,
  `*.finished.json`은종료코드·timeout 여부다. 모델`run.json`과`progress.json`으로
  실제첫갱신 여부를 확인한다. tmux 시작만으로 학습이 시작됐다고 세지 않는다.
- 학습3시간, 평가각1시간, GPU메모리부족 대기최대8시간. 다른 사용자의 프로세스 미변경.
  실패 시 부분모델은 삭제하지 않고INCOMPLETE.json으로 격리, 후속 단계는 중단한다.
- `scripts/run_low_noise_student_lecun.py` 추가. 실행계약 및기존GPU대기 검사7개통과,
  Ruff통과. 기존 구현76테스트와DINO사전검사는 [준비 기록](../x0-loss-audit/README.md).
- 기존 모델·데이터·물리 설정은 보존한다. Claude Milk/multiccd는별도.

## 실제 시작 확인

사전 검사·기록/정지 대조군·DINO 검사가 통과한 뒤 **22:31:30 KST에 첫 optimizer 갱신**이
실행됐다. `run.json`의 training_started=true를 확인했고, 첫 확인은80/6,331updates,
유한 weighted train_loss0.17687이었다. 초기속도0.53초/update로 학습약1시간+평가시간을
예상하지만 초기 추정이며 완료 시각 보장은 아니다. A와 loss 숫자를 직접 비교하지 않는다.

[startup-verification.json](startup-verification.json)은 새 run과 A의 normalizer·dataset·cache·
DINO·joint/control 계약 일치와 low_noise_loss만 다른 설정임을 확인한다.
로컬 `run.json`·`progress.json`은 시작 시점 사본으로 실시간 상태나 완료 결과가 아니다.
학습과 자동 평가는 원격 tmux에서 계속되며 네트워크가 끊겨도 독립 실행된다.

## 병행 teacher 작업에 대한 확인

Claude Milk B의 기존1.2초 gate 통과와 마지막0.5초 지속 안정화 실패는 별개다.
이번 teacher 진행조건은1.2초 gate로 명시돼 있으므로 제한 예산 탐색을 계속할 수 있지만,
지속 안정화 실패와 brace 조건 차이는 결과/HANDOFF에 남겨야 한다. B 성공을 기존A
student 데이터에 자동으로 합치거나 충분한 안정화가 확인됐다고 쓰지 않는다.
이 학습은 기존A 단일 병 자료만 사용하며 Milk/B는 들어가지 않는다.
