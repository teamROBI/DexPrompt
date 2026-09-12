#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="$1"
TEACHER_PY=third_party/dexonomy/.venv/bin/python
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN_ROOT/launcher-exit.json"' EXIT
timeout --signal=TERM --kill-after=30s 15m nice -n 15 "$TEACHER_PY" \
    scripts/data/prepare_followup_intake.py \
    --source /data1/yspark/sim/icra/teacher-followup-5h-20260912-v1 \
    --out "$RUN_ROOT/preparation"
nice -n 15 "$TEACHER_PY" scripts/data/intake_hope_diversity.py \
    --frozen-plan "$RUN_ROOT/preparation/intake-plan.json" \
    --out "$RUN_ROOT/intake" --renderer egl --egl-device 0 --hours 3
printf '%s\n' 'INTAKE RUNNER DONE (see results.json for failures and pending episodes)'
