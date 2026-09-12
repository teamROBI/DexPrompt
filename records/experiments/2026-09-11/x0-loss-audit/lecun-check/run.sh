#!/bin/bash
set -u
cd /home/yspark/projects/DexPrompt/out/student-x0-low-noise-code-v1
job=/data1/yspark/projects/dexprompt/student-checks/x0-low-noise-check-job-v1
date --iso-8601=seconds > "$job/started-at"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=GPU-432d6dc7-116e-49f9-8eae-badd48341b90 timeout --signal=TERM --kill-after=15s 1800 nice -n 15 /home/yspark/projects/DexPrompt/.venv-student/bin/python "$job/check.py" > "$job/run.log" 2>&1
code=$?
printf '%s\n' "$code" > "$job/exit-code"
date --iso-8601=seconds > "$job/finished-at"
exit "$code"
