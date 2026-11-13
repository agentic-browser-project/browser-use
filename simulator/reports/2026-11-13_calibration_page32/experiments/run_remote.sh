#!/bin/bash
set -eu
work=/home/shiqihe/workspace/calibration_page32_20261113
exec 9>"$work/experiment.lock"
flock -w 30 9 || exit 1
exec docker exec -e MAX_JOBS=2 browsa-score-error-20261111 \
 /workspace/anchor_alpha_grid_20261110/.venv/bin/python \
 /workspace/calibration_page32_20261113/experiments/bootstrap.py
