#!/bin/bash
set -u
cd /home/yspark/projects/DexPrompt/out/student-x0-low-noise-code-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yspark/projects/DexPrompt/.venv-student/bin/python scripts/run_low_noise_student_lecun.py > launch.log 2>&1
code=$?
printf '%s\n' "$code" > launch.exit
exit "$code"
