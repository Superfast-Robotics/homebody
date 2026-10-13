#!/usr/bin/env bash
# run_jobs.sh BASE PARALLEL JOBS_FILE EVOLVE_DIR MODEL: run every "CANDIDATE SCENE TARGETS OUT" line of
# JOBS_FILE as one HomeBody session (the agent may customize skills) with the candidate's code,
# then score it from physics. Astra is reached through the Codex relay.
set -uo pipefail
BASE=$1; PARALLEL=$2; JOBS=$3; EVOLVE=$4; MODEL=$5
W=$(dirname "$BASE")
run_one() {
  local cand=$1 scene=$2 targets=$3 out=$4
  mkdir -p "$out"
  local task
  task=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['default_task'])" "$scene/task_scene.json")
  PATH="$W/relay-bin:$PATH" CODEX_RELAY_TOKEN="$(cat ~/.codex_relay_token)" MUJOCO_GL=egl \
    PYTHONPATH="$cand/src:$cand/vendor" timeout 7200 "$BASE/.venv/bin/python" -m homebody --headless --root "$cand" \
    --scene "$scene" --model "$MODEL" --settings "$EVOLVE/episode_settings.toml" --output "$out/session" \
    --task "$task" > "$out/run.log" 2>&1
  echo "$?" > "$out/exit_code"
  "$BASE/.venv/bin/python" "$EVOLVE/score_episode.py" "$out/session" "$targets" "$cand/evaluation/evaluate.py" \
    > "$out/result.json" 2> "$out/score.err" || rm -f "$out/result.json"
}
export -f run_one
export BASE W EVOLVE MODEL
xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$JOBS"
