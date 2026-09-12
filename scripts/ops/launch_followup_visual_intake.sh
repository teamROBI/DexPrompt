#!/usr/bin/env bash
# Fresh diagnostic intake; does not train or modify the teacher/student pools.
set -euo pipefail
cd /home/yspark/projects/ICRA
RUN_ROOT="${1:-/data1/yspark/sim/icra/followup-visual-intake-20260912-v1}"
SESSION_NAME=codex-followup-intake
if [[ -e "$RUN_ROOT" ]] || tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
    printf '%s\n' 'Existing output/session: refusing to start a duplicate job.' >&2
    exit 2
fi
mkdir -p "$RUN_ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1
export TMPDIR="$RUN_ROOT/tmp"
mkdir "$TMPDIR"
printf -v OUT_QUOTED '%q' "$RUN_ROOT"
tmux new-session -d -s "$SESSION_NAME" -c "$PWD" \
    "exec bash scripts/ops/run_followup_visual_intake.sh $OUT_QUOTED > $OUT_QUOTED/launcher.log 2>&1"
printf 'session=%s\nout=%s\n' "$SESSION_NAME" "$RUN_ROOT"
