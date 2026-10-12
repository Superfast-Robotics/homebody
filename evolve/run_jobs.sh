#!/usr/bin/env bash
# run_jobs.sh BASE PARALLEL JOBS_FILE WALL_TIMEOUT: run every "CANDIDATE SCENE SEQUENCE OUT" line of
# JOBS_FILE as one scripted check with the candidate's code on the base checkout's interpreter.
set -uo pipefail
BASE=$1; PARALLEL=$2; JOBS=$3; WALL=$4
run_one() {
  local cand=$1 scene=$2 sequence=$3 out=$4
  mkdir -p "$out"
  MUJOCO_GL=egl PYTHONPATH="$cand/src:$cand/vendor" timeout $((WALL + 120)) \
    "$BASE/.venv/bin/python" "$cand/tools/check_manipulation.py" --root "$cand" --scene "$scene" \
    --sequence "$sequence" --no-visuals --wall-timeout "$WALL" --output "$out" > "$out/run.log" 2>&1
  echo "$?" > "$out/exit_code"
}
export -f run_one
export BASE WALL
xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$JOBS"
