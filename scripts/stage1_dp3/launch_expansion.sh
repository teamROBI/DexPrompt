#!/usr/bin/env bash
set -euo pipefail
DP3_JOB=/data1/yspark/projects/dexprompt/stage1-dp3-placement-v1
DP3_CODE="$DP3_JOB/package/code"
export CUDA_VISIBLE_DEVICES=3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DP3_CODE/scripts:$DP3_CODE/src:$DP3_CODE/vendor:/data1/yspark/projects/dexprompt/stage1-dp3-v1/deps:/data1/yspark/projects/dexprompt/student-code/upstream-action-v1/deps"
DP3_PYTHON=/home/yspark/projects/DexPrompt/.venv-student/bin/python
cd "$DP3_JOB"
trap 'code=$?; echo "$code" > launcher-exit.txt' EXIT
"$DP3_PYTHON" -c 'import subprocess; free=int(subprocess.check_output(["nvidia-smi","-i","3","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip()); assert free > 12000, free'
nice -n 10 "$DP3_PYTHON" -m stage1_dp3.experiment check --package "$DP3_JOB/package" --out "$DP3_JOB/model-check.json"
nice -n 10 "$DP3_PYTHON" -m stage1_dp3.expansion run --package "$DP3_JOB/package" --out "$DP3_JOB/run"
