#!/usr/bin/env bash
# run_jobs.sh BASE PARALLEL JOBS_FILE EVOLVE_DIR MODEL [SETTINGS [STOP_FILE [BUDGET]]]: run every
# "CANDIDATE SCENE TARGETS OUT" line of JOBS_FILE as one HomeBody session (the agent may customize skills) with the
# candidate's code, then score it from physics with the loop's own evaluator (EVOLVE_DIR/evaluate.py, never the
# candidate's copy). The agent model is reached through the LiteLLM proxy tunnel (port 14000) and the coding
# sub-agent through the Codex relay.
# Each OUT gets usage.json (the session's credits, whether or not it scores), result.json (only when scoring
# succeeded; the scorer's own output stays in score.out and score.err) and exit_code. With STOP_FILE, a session
# that hit the model's usage limit, or credits charged (STOP_FILE.ledger, shared by every host of the batch)
# reaching BUDGET, creates STOP_FILE: the sessions not yet started are skipped (OUT/skipped) for a later resume.
set -uo pipefail
# Episodes start together and share the candidate's folder on the group volume: concurrent bytecode-cache
# writes and renames there deadlocked a host's mount (virtiofs), so no episode writes __pycache__.
export PYTHONDONTWRITEBYTECODE=1
BASE=$1; PARALLEL=$2; JOBS=$3; EVOLVE=$4; MODEL=$5; SETTINGS=${6:-$4/episode_settings.toml}
STOP=${7:-}; BUDGET=${8:-0}
W=$(dirname "$BASE")
QUOTA='usage limit|rate limit exceeded|insufficient_quota'
run_one() {
  local cand=$1 scene=$2 targets=$3 out=$4
  mkdir -p "$out"
  if [ -n "$STOP" ] && [ -e "$STOP" ]; then echo "stopped: $(cat "$STOP")" > "$out/skipped"; return; fi
  rm -f "$out/skipped"
  local task
  task=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['default_task'])" "$scene/task_scene.json")
  PATH="$W/relay-bin:$PATH" CODEX_RELAY_TOKEN="$(cat ~/.codex_relay_token)" MUJOCO_GL=egl \
    LITELLM_BASE_URL=http://127.0.0.1:14000 LITELLM_API_KEY="$(cat ~/.litellm_key)" HOMEBODY_CODER_CODEX="$W/coder-codex/codex" \
    PYTHONPATH="$cand/src:$cand/vendor" timeout 7200 "$BASE/.venv/bin/python" -m homebody --headless --root "$cand" \
    --scene "$scene" --model "$MODEL" --settings "$SETTINGS" --output "$out/session" \
    --task "$task" > "$out/run.log" 2>&1
  echo "$?" > "$out/exit_code"
  "$BASE/.venv/bin/python" "$EVOLVE/score_episode.py" --usage "$out/session" "$MODEL" > "$out/usage.json" 2>> "$out/score.err" \
    || echo '{"credits": 0.0, "unmetered": true}' > "$out/usage.json"
  rm -f "$out/result.json"
  "$BASE/.venv/bin/python" "$EVOLVE/score_episode.py" "$out/session" "$targets" "$EVOLVE/evaluate.py" "$MODEL" \
    > "$out/score.out" 2>> "$out/score.err" && cp "$out/score.out" "$out/result.json"
  [ -n "$STOP" ] || return 0
  if grep -qiE "$QUOTA" "$out/run.log" "$out/session"/*/tasks/*/outcome.json 2>/dev/null; then
    echo "the model's usage limit was hit in $out" > "$STOP"
  fi
  python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('credits',0))" "$out/usage.json" >> "$STOP.ledger" 2>/dev/null
  if [ "$BUDGET" != "0" ] && python3 -c "import sys;sys.exit(0 if sum(float(x) for x in open(sys.argv[1]) if x.strip()) >= float(sys.argv[2]) else 1)" \
      "$STOP.ledger" "$BUDGET"; then
    echo "the evaluation's credit budget ($BUDGET) is spent" > "$STOP"
  fi
}
export -f run_one
export BASE W EVOLVE MODEL SETTINGS STOP BUDGET QUOTA
xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$JOBS"
