#!/bin/bash
set -u
cd /home/yspark/projects/DexPrompt/out/student-x0-code-v1
job=/data1/yspark/projects/dexprompt/student-checks/teacher-observations-50-job-v1
date --iso-8601=seconds > "$job/started-at"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=GPU-432d6dc7-116e-49f9-8eae-badd48341b90 timeout --signal=TERM --kill-after=15s 1800 nice -n 15 /home/yspark/projects/DexPrompt/.venv-student/bin/python scripts/diagnose_teacher_observations_50.py --plan "$job/plan.json" --out /data1/yspark/projects/dexprompt/student-checks/teacher-observations-50-v1 > "$job/run.log" 2>&1
code=$?
printf '%s\n' "$code" > "$job/exit-code"
date --iso-8601=seconds > "$job/finished-at"
exit "$code"
