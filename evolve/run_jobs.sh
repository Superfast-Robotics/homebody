#!/usr/bin/env bash
# run_jobs.sh BASE PARALLEL JOBS_FILE ATTEMPTS EFFORT INNER: run every "CANDIDATE SCENE TARGETS OUT" line
# of JOBS_FILE as one inner loop (INNER, evolve/inner.py) with the candidate's code; astra is reached
# through the Codex relay.
set -uo pipefail
BASE=$1; PARALLEL=$2; JOBS=$3; ATTEMPTS=$4; EFFORT=$5; INNER=$6
W=$(dirname "$BASE")
run_one() {
  local cand=$1 scene=$2 targets=$3 out=$4
  mkdir -p "$out"
  PATH="$W/relay-bin:$PATH" CODEX_RELAY_TOKEN="$(cat ~/.codex_relay_token)" MUJOCO_GL=egl \
    PYTHONPATH="$cand/src:$cand/vendor" timeout 5400 "$BASE/.venv/bin/python" "$INNER" --root "$cand" \
    --scene "$scene" --targets "$targets" --out "$out" --attempts "$ATTEMPTS" --effort "$EFFORT" > "$out/run.log" 2>&1
  echo "$?" > "$out/exit_code"
}
export -f run_one
export BASE W INNER ATTEMPTS EFFORT
xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$JOBS"
