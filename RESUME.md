# 현재 작업 상태 — 2026-09-10

## 새 서버로 이동 준비

사용자가 `https://github.com/teamROBI/DexPrompt.git` 업로드와 3090 서버 이전 준비를 요청했다.
설치·전송·검사·tmux 실행 명령은 [SERVER_3090.md](SERVER_3090.md)에 고정했다.
전송 묶음은 `/data1/yspark/sim/icra/student-transfer/bottle-can-eval-v1.tar`이며
SHA256은 `66ffda013302d5816d8d2d9ac314f9ef71725d02b5237c793e4e0668047bf1f7`이다.
기존 서버에서 새 학습이나 학습 정책 평가를 재시작하지 않는다.
새 서버의 다음 작업은 최종 epoch_0002 체크포인트의 4개 과제 × 3회 평가다.
GitHub push는 아직 미완료다: 서버 Git 인증이 없고 연결 앱의 저장소 권한도 `push: false`다.
업로드용 별도 Git 저장소는 `out/github-publish/DexPrompt`, 오프라인 clone용 묶음은
`out/github-publish/DexPrompt.bundle`이다. 쓰기 인증 후 별도 저장소에서 `git push -u origin main`을 실행한다.
원래 프로젝트의 `.git`/index는 변경하지 않았다.

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

상세 기록: [학습 실행 보고서](results/2026-09-10/visual-student-bc-v1/README.md).

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
