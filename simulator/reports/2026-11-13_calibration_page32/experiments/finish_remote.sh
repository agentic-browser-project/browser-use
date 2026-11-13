#!/bin/bash
set -eu
work=/home/shiqihe/workspace/calibration_page32_20261113
exec 8>"$work/postprocess.lock"
flock -n 8 || exit 0
last_count=0
while true; do
    count=0
    if test -f "$work/progress.json"; then
        count=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["completed_tasks"])' "$work/progress.json")
    fi
    if test "$count" -ge 16 && test "$count" -gt "$last_count"; then
        docker exec browsa-score-error-20261111 /workspace/anchor_alpha_grid_20261110/.venv/bin/python \
            /workspace/calibration_page32_20261113/experiments/analyze_replay.py
        last_count=$count
    fi
    if test "$count" -eq 64; then
        date -u > "$work/postprocess_complete.txt"
        exit 0
    fi
    if ! pgrep -f '^/workspace/anchor_alpha_grid_20261110/.venv/bin/python /workspace/calibration_page32_20261113/experiments/bootstrap.py$' > /dev/null; then
        echo 'GPU replay stopped before 64 tasks. Inspect replay.log and control checks.'
        exit 1
    fi
    sleep 30
done
