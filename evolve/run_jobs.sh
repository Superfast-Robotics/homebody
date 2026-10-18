#!/usr/bin/env bash
# run_jobs.sh BASE PARALLEL JOBS_FILE EVOLVE_DIR MODEL [PYTHON] [ROBOT]: run every "CANDIDATE SCENE TARGETS OUT"
# line of JOBS_FILE as one HomeBody session (the agent may customize skills) with the candidate's code, on
# PYTHON's environment (BASE's by default) and --robot.type ROBOT (mujoco by default), then score it from
# physics. The agent model is reached through the LiteLLM proxy tunnel (port 14000) and the coding sub-agent
# through the Codex relay.
set -uo pipefail
BASE=$1; PARALLEL=$2; JOBS=$3; EVOLVE=$4; MODEL=$5; PYTHON=${6:-$BASE/.venv/bin/python}; ROBOT=${7:-mujoco}
W=$(dirname "$BASE")
run_one() {
  local cand=$1 scene=$2 targets=$3 out=$4
  mkdir -p "$out"
  local task robot=()
  [ "$ROBOT" = mujoco ] || robot=(--robot.type "$ROBOT")  # code from before --robot.type existed runs as ever
  task=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['default_task'])" "$scene/task_scene.json")
  PATH="$W/relay-bin:$PATH" CODEX_RELAY_TOKEN="$(cat ~/.codex_relay_token)" MUJOCO_GL=egl \
    LITELLM_BASE_URL=http://127.0.0.1:14000 LITELLM_API_KEY="$(cat ~/.litellm_key)" HOMEBODY_CODER_CODEX="$W/coder-codex/codex" \
    PYTHONPATH="$cand/src:$cand/vendor:$cand/packages/lerobot_robot_homebody_sim" timeout 7200 "$PYTHON" -m homebody \
    --headless --root "$cand" "${robot[@]}" \
    --scene "$scene" --model "$MODEL" --settings "$EVOLVE/episode_settings.toml" --output "$out/session" \
    --task "$task" > "$out/run.log" 2>&1
  echo "$?" > "$out/exit_code"
  "$BASE/.venv/bin/python" "$EVOLVE/score_episode.py" "$out/session" "$targets" "$cand/evaluation/evaluate.py" "$MODEL" \
    > "$out/result.json" 2> "$out/score.err" || rm -f "$out/result.json"
}
export -f run_one
export BASE W EVOLVE MODEL PYTHON ROBOT
xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$JOBS"
