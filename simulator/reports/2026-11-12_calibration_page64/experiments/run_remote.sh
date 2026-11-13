#!/bin/bash
set -eu
work=/home/shiqihe/workspace/calibration_page64_20261112
exec 9>"$work/experiment.lock"
flock -w 30 9 || { echo 'This experiment is already running'; exit 1; }
exec docker exec -e MAX_JOBS=2 browsa-score-error-20261111 \
    /workspace/anchor_alpha_grid_20261110/.venv/bin/python \
    /workspace/calibration_page64_20261112/experiments/run_replay.py
