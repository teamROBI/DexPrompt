# RTX 3090 서버로 이동하기

현재 가장 먼저 할 일은 **이미 학습한 시각 student의 폐루프 평가**입니다.
병 3개 시범 + 캔 1개 시범으로 3 epoch, 6,255 updates를 마쳤습니다.
최종 `epoch_0002.pt`를 사전에 고정했으며, 아직 최종 파지 성공률은 없습니다.
이 안내의 `start`는 평가만 실행합니다. 학습을 다시 시작하지 않습니다.

## 1. 코드와 환경

```bash
git clone https://github.com/teamROBI/DexPrompt.git
cd DexPrompt
bash scripts/setup_student_server.sh
```

GitHub 업로드가 인증 때문에 아직 대기 중이거나 GitHub 접속이 어려우면,
기존 서버의 `/home/yspark/projects/ICRA/out/github-publish/DexPrompt.bundle`을
새 서버로 복사하고 `git clone DexPrompt.bundle DexPrompt`로 같은 코드 스냅샷을 받을 수 있습니다.
그 뒤 환경 설치와 아래 6.02GB 실험 파일 전송 절차는 같습니다.

Linux, Python 3.10과 venv, `tmux`, `nvidia-smi`, `nice`, `ionice`가 필요합니다.
렌더링에는 EGL/OpenGL 라이브러리, 아래 CPU 대조 검사에는 OSMesa가 필요합니다.
Ubuntu 계열 패키지 이름은 `python3.10-venv`, `tmux`, `libegl1`, `libgl1`,
`libosmesa6`입니다. 시스템 패키지나 NVIDIA 드라이버 설치가 필요하면 서버 관리자에게 요청하세요.
설치 스크립트는 프로젝트의 `.venv-student`에만 Python 패키지를 설치합니다.
Python 3.10이 다른 이름이라면 `STUDENT_PYTHON=/path/to/python bash scripts/setup_student_server.sh`로 실행합니다.

검증된 버전은 torch 2.6.0 / torchvision 0.21.0의 CUDA 12.4 wheel,
MuJoCo 3.3.3입니다. CUDA Toolkit을 별도로 컴파일해 사용하지 않습니다.
실제 서버의 드라이버 호환성과 EGL 렌더링은 실행 전 검사합니다.
MuJoCo의 저장된 MJB를 쓰므로 버전을 임의로 변경하지 마세요.

## 2. 실험 파일 전송

Git에는 코드·설정·테스트·보고서가 들어갑니다. 용량이 큰 시범과 가중치는
Git에 넣지 않았습니다. **clone 후 아래 묶음도 한 번 전송해야 합니다.**

기존 서버에 준비된 파일:

```text
/data1/yspark/sim/icra/student-transfer/bottle-can-eval-v1.tar
/data1/yspark/sim/icra/student-transfer/bottle-can-eval-v1.tar.sha256
```

새 서버의 저장 공간이 넉넉한 디렉터리에서 실행합니다. `SOURCE_HOST`는 기존 서버의 SSH 주소로 바꿉니다.
코드·환경·압축 전송본·해제본을 합쳐 최소 25GB 이상의 여유를 권합니다.

```bash
mkdir -p ../dexprompt-transfer
rsync -avP yspark@SOURCE_HOST:/data1/yspark/sim/icra/student-transfer/bottle-can-eval-v1.tar ../dexprompt-transfer/
rsync -avP yspark@SOURCE_HOST:/data1/yspark/sim/icra/student-transfer/bottle-can-eval-v1.tar.sha256 ../dexprompt-transfer/
(cd ../dexprompt-transfer && sha256sum -c bottle-can-eval-v1.tar.sha256)
tar -xf ../dexprompt-transfer/bottle-can-eval-v1.tar -C .
```

묶음은 약 6GB이며 다음 경로로 풀립니다. 중간 epoch, 학습 캐시, 과거 실패 실험은 포함하지 않았습니다.

| 경로 | 내용 |
| --- | --- |
| `out/student-bottle-can-v1/` | 4개 시범, 이미지, 봉인된 manifest, 재현용 MJB와 초기 상태 |
| `out/student-assets/dinov2/` | 학습에 사용한 DINO 소스 및 라이선스 |
| `out/student-assets/*pretrain.pth` | head ViT-B/14, wrist ViT-L/14 가중치 |
| `out/student-checkpoints/bottle-can-bc-v1/epoch_0002.pt` | 사전 고정한 최종 체크포인트 |
| `out/student-transfer/manifest.json` | 파일별 크기·SHA256·상대 경로 |

이 **저장된 4개 과제의 평가**에는 별도 Dexonomy checkout이나 원래 DGN 물체 폴더가 필요하지 않습니다.
새 teacher 후보 생성·새 물체 추가에는 추가 자산이 필요하며 [SETUP.md](SETUP.md)의 teacher 설정을 따릅니다.
전송 파일은 해당 프로젝트 실험에 사용하고 포함된 외부 자산의 원래 이용 조건을 유지합니다.

## 3. 전송 무결성과 물리 재현 검사

```bash
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-student/bin/python scripts/student_server.py verify --load-model

CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-student/bin/python scripts/evaluate_student_suite.py \
  --data out/student-bottle-can-v1 --out out/server-controls \
  --check-controls --train-task-diagnostic
```

첫 검사는 파일·학습 소스 해시, 데이터 계약, 실제 DINO 포함 체크포인트 로드를 확인합니다.
CPU에서 로드만 하며 학습이나 정책 추론은 하지 않습니다.
둘째 검사는 기록된 행동과 제자리 유지 대조군을 다시 실행합니다.
`out/server-controls/summary.json`의 `completed`와 `checks_passed`가 모두 true여야 합니다.
기대값은 기록된 행동 4/4 완주, 제자리 유지 0/4입니다.
오류가 나면 원인을 먼저 해결하고 새 출력 경로로 재검사합니다. 해시 검사를 우회하지 마세요.

## 4. GPU 한 장에서 짧게 실행

```bash
nvidia-smi
.venv-student/bin/python scripts/student_server.py start \
  --gpu 0 --smoke --repeats 1 --out out/eval3090-smoke --session dexprompt-smoke
```

`--gpu`는 `nvidia-smi`에 표시되는 물리 GPU 번호입니다. CUDA에는 그 GPU의 UUID만 노출하며
프로세스 안에서는 `cuda:0`이 됩니다. 여유 메모리 12,000 MiB 미만이거나 이용률 20% 초과이면
시작을 거절합니다. 이는 시작 시 확인이며 자원 예약·메모리 상한은 아닙니다.
다른 사용자가 쓸 GPU는 고르지 마세요. 6장을 동시에 채우는 방식은 기본값이 아닙니다.

실행기는 CUDA 계산과 작은 EGL 렌더를 먼저 검사하고, 실패하면 정책 평가 전에 중지합니다.
기본 EGL 장치 번호도 `--gpu`와 같습니다. 컨테이너 등에서 EGL 장치 순서가 다르면
렌더링 장치 매핑을 먼저 확인하고 `--egl-device N`으로 따로 지정해야 합니다.
BLAS 스레드는 1개, CPU nice는 10, I/O는 idle 우선순위로 실행합니다.

```bash
tmux attach -t dexprompt-smoke
cat out/eval3090-smoke/status.json
tail -f out/eval3090-smoke/evaluation.log
```

자세한 episode 로그는 `out/eval3090-smoke/results/*.log`입니다.
`Ctrl-b`를 누른 뒤 `d`로 tmux를 분리합니다. SSH가 끊겨도 계속 실행됩니다.
완료되면 tmux 세션은 사라지며 `status.json`에 종료 코드가 남습니다.
`state: completed`는 실행 종료를 뜻합니다. **파지 성공 여부는 `results/summary.json`의
`compliant_successes`를 봐야 합니다.** 첫 시험에서 파지가 실패해도 실행 오류가 없다면 본 평가로 진행합니다.

## 5. 고정한 전체 평가: 시범 4개 × seed 3개

짧은 실행이 정상 종료한 뒤, 같은 여유 GPU에서 실행합니다.

```bash
.venv-student/bin/python scripts/student_server.py start \
  --gpu 0 --out out/eval3090-final --session dexprompt-final
```

총 12회, 각 과제의 seed 1000·1001·1002, 같은 최종 체크포인트와 성공 조건을 사용합니다.
출력 경로는 매번 새것이어야 합니다. 중단한 실행에 자동으로 이어 쓰지 않습니다.
GPU 평가는 CPU에서 중단했던 결과와 합치지 않습니다.

확인할 결과는 `out/eval3090-final/results/summary.json`과 episode별 `evaluation.json`입니다.
완주·필수 접촉 유지·품질 위반, 물체별 성공률과 전체 12회 분모를 함께 보고합니다.
이는 **학습에 쓴 고정 과제 진단**입니다. 새로운 물체나 배치에 대한 일반화 결과로 해석하지 않습니다.
3090에서의 실제 처리 시간과 메모리는 첫 실행 후 기록합니다. 현재 서버에서 이를 측정했다고 주장하지 않습니다.

다음 실험은 이 결과로 정합니다. 0회 성공이면 시각 정책의 폐루프 오차를 먼저 진단하고,
성공이 있으면 새 배치·시범 다양화와 별도 평가 세트를 준비합니다.
지금 GPU가 6장 있다는 이유로 같은 데이터를 더 오래 재학습하지 않습니다.
