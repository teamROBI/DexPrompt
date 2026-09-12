#!/usr/bin/env bash
# Launch only after Claude implements/tests run_teacher_followup_5h.py.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${1:-/data1/yspark/sim/icra/teacher-followup-5h-20260912-v1}"
SESSION_NAME="icra-followup-5h"
PYTHON_BIN="$PROJECT_ROOT/third_party/dexonomy/.venv/bin/python"
RUNNER="$PROJECT_ROOT/scripts/experiments/run_teacher_followup_5h.py"

if [[ ! -f "$RUNNER" ]]; then
    printf '%s\n' '먼저 records/plans/CLAUDE_FIVE_HOURS_2026-09-12.md에 따라 실행기를 구현·검사해야 합니다.' >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf '%s\n' "Python 실행 파일 없음: $PYTHON_BIN" >&2
    exit 2
fi
if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
    printf '%s\n' "이미 실행 중인 tmux: $SESSION_NAME" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT/launcher.log" ]]; then
    printf '%s\n' '기존 실행의 launcher.log가 있습니다. 새 경로 또는 검증된 resume 절차를 사용하세요.' >&2
    exit 2
fi

mkdir -p "$RUN_ROOT"
COMMAND=(timeout --signal=TERM --kill-after=60s 5h
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    MUJOCO_GL=egl PYTHONUNBUFFERED=1
    nice -n 15 "$PYTHON_BIN" "$RUNNER"
    --out "$RUN_ROOT" --hours 5 --workers 2 --max-physics-runs 163)
printf -v SHELL_COMMAND '%q ' "${COMMAND[@]}"
printf -v LOG_PATH '%q' "$RUN_ROOT/launcher.log"
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT" \
    "exec $SHELL_COMMAND > $LOG_PATH 2>&1"
printf 'tmux: %s\noutput: %s\n' "$SESSION_NAME" "$RUN_ROOT"
