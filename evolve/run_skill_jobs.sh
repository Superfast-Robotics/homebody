#!/usr/bin/env bash
# run_skill_jobs.sh BASE PARALLEL JOBS_FILE EVOLVE_DIR: run every "CANDIDATE EPISODE OUT" line of JOBS_FILE
# as one skill-benchmark episode (no model) with the candidate's code.
set -uo pipefail
BASE=$1; PARALLEL=$2; JOBS=$3; EVOLVE=$4
run_one() {
  local cand=$1 episode=$2 out=$3
  mkdir -p "$out"
  MUJOCO_GL=egl timeout 1800 "$BASE/.venv/bin/python" "$EVOLVE/skill_bench.py" "$cand" "$episode" "$out" \
    > "$out/run.log" 2>&1
  echo "$?" > "$out/exit_code"
}
export -f run_one
export BASE EVOLVE
xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$JOBS"
