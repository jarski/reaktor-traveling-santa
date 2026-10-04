#!/bin/sh
# Run santa2 with the configuration used for the leaderboard result, then
# validate the output with the independent scorer.
#   ./run.sh <seconds> [seed.csv]
# santa2 rewrites out/best.csv on every improvement, so the run can be
# interrupted at any point and the file on disk is a complete, valid solution.
set -eu
cd "$(dirname "$0")"
SECS=${1:-600}
SEED=${2:-}
THREADS=${THREADS:-$(getconf _NPROCESSORS_ONLN)}
mkdir -p out
SANTA_T0=${SANTA_T0:-500} SANTA_T1=${SANTA_T1:-1.0} SANTA_ROUND=${SANTA_ROUND:-10} \
    ./santa2 "$SECS" "$THREADS" out/best.csv $SEED
python3 ../tools/score.py out/best.csv
