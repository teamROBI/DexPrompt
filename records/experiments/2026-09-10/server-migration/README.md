<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 새 서버용 GitHub 코드와 평가 자산 준비

2026-09-10. 대상 저장소는 `https://github.com/teamROBI/DexPrompt.git`.
실행 순서는 루트 [SERVER_3090.md](../../../../docs/lecun.md)에 고정했다.
현재 작업트리의 코드를 별도 스냅샷으로 게시해 기존 로컬 브랜치·index·진행 중 변경을 보존한다.
기존 서버에서 새 학습이나 학습된 정책의 폐루프 평가는 시작하지 않았다.

**GitHub 업로드는 인증 대기 중이다.** HTTPS push는 인증 정보가 없어 실패했고,
연결 GitHub 앱의 저장소 권한도 `pull: true`, `push: false`로 확인했다.
업로드용 로컬 Git 저장소와 bundle을 `out/github-publish/`에 보존한다.

## 전송 파일

- `/data1/yspark/sim/icra/student-transfer/bottle-can-eval-v1.tar`
- 6,021,109,760 bytes (6.02 GB / 5.61 GiB), 258개 파일.
- archive SHA256: `66ffda013302d5816d8d2d9ac314f9ef71725d02b5237c793e4e0668047bf1f7`.
- 최종 checkpoint SHA256: `8d6ec064fa6e528f4699d63a2804b40ab376756d4e622c5f31ac2bdc7364550b`.
- dataset manifest SHA256: `267c23e2fa9a7d4104aa37d8b7436457301c4828d1af92c207f2ee986607946f`.
- 파일별 명세: [artifacts.json](artifacts.json).
- Git에는 모델 가중치·시범 묶음을 올리지 않는다. 원본 가중치, 시범, 체크포인트는 변경하지 않았다.

## 경로 분리 검증

게시할 코드와 위 tar만 다음 별도 디렉터리에 복사·해제했다:
`/data1/yspark/sim/icra/student-transfer/clean-check/DexPrompt`.
이 디렉터리에는 `third_party/dexonomy`, 기존 DGN object 폴더, `data` 심볼릭 링크가 없다.
Python 환경은 기존 `.venv-student` 인터프리터를 재사용했다. 새 서버에서 pip 설치까지
실행한 검증은 아니며, 설치 스크립트에는 실제 사용 중인 패키지 버전을 고정했다.

- `student_server.py verify --load-model`: 258개 파일 해시 일치, dataset 계약 통과,
  6,255 updates / epoch 2 확인, 학습 소스 해시 일치, 실제 DINO 포함 체크포인트 CPU 로드 통과.
- `evaluate_student_suite.py --check-controls`: 기록된 행동 4/4 teacher-compliant lift,
  제자리 유지 0/4 sustained lift. `completed: true`, `checks_passed: true`.
  원시 기록: [controls.json](controls.json).
- `replay_hx5_student.py --check-reset-only`: 병 1개 초기 상태의 CPU 렌더 통과,
  head RGB+mask 256×256×4 / wrist RGB 256×256×3 / 측정 관절 27차원 확인.
- 새 서버 도구 및 기존 student 준비·추론 회귀 테스트 **31 passed**.
- 변경 Python 파일 Ruff, 설치 스크립트 `bash -n`, `git diff --check` 통과.

이 검사는 학습 정책의 파지 성공률을 측정하지 않았다. 3090 GPU의 메모리·처리량·EGL은
새 서버의 짧은 실행에서 확인해야 한다. PyTorch CUDA 계산과 EGL 렌더 검사 실패 시
평가 이전에 종료하도록 연결했다.

## 다음 실행

1. 새 서버에서 clone, 환경 설치, tar 전송·해제, `verify --load-model`.
2. CPU 기록 행동 대조 검사 통과 확인.
3. 빈 GPU 한 장에서 `start --smoke --repeats 1`.
4. 실행 오류가 없으면 고정한 epoch 2, 4개 과제 × seed 1000–1002의 본 평가.
5. train-task 진단으로 결과를 보고한다. 현재 CPU 중단 기록이나 새 물체 일반화와 혼합하지 않는다.
