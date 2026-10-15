#!/usr/bin/env bash
# run_skill_jobs.sh BASE PARALLEL JOBS_FILE EVOLVE_DIR [CODER [CODER_MODEL [CODER_RATES]]]: run every "CANDIDATE EPISODE OUT" line of
# JOBS_FILE as one skill-benchmark episode with the candidate's code; CODER=1 lets a failed skill call
# the coding assistant once (through the LiteLLM tunnel and the remote Codex).
set -uo pipefail
BASE=$1; PARALLEL=$2; JOBS=$3; EVOLVE=$4; CODER=${5:-0}; CODER_MODEL=${6:-}; CODER_RATES=${7:-}
W=$(dirname "$BASE")
run_one() {
  local cand=$1 episode=$2 out=$3
  mkdir -p "$out"
  MUJOCO_GL=egl HOMEBODY_BENCH_CODER="$CODER" HOMEBODY_CODER_CODEX="$W/coder-codex/codex" \
    HOMEBODY_CODER_MODEL="$CODER_MODEL" HOMEBODY_CODER_RATES="$CODER_RATES" \
    LITELLM_BASE_URL=http://127.0.0.1:14000 LITELLM_API_KEY="$(cat ~/.litellm_key)" \
    timeout 1800 "$BASE/.venv/bin/python" "$EVOLVE/skill_bench.py" "$cand" "$episode" "$out" \
    > "$out/run.log" 2>&1
  echo "$?" > "$out/exit_code"
}
export -f run_one
export BASE EVOLVE CODER CODER_MODEL CODER_RATES W
xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$JOBS"
