#!/usr/bin/env bash
# run_jobs.sh BASE PARALLEL JOBS_FILE EVOLVE_DIR MODEL [SETTINGS [STOP_FILE [BUDGET RESERVE LEDGER]]]: run every
# "CANDIDATE SCENE TARGETS OUT" line of JOBS_FILE as one HomeBody session (the agent may customize skills) with the
# candidate's code, then score it from physics with the loop's own evaluator (EVOLVE_DIR/evaluate.py, never the
# candidate's copy). The agent model is reached through the LiteLLM proxy tunnel (port 14000) and the coding
# sub-agent through the Codex relay.
# Each OUT gets usage.json (the session's credits, whether or not it scores), result.json (only when scoring
# succeeded; the scorer's own output stays in score.out and score.err) and exit_code. With STOP_FILE, a session
# that hit the model's usage limit creates STOP_FILE: the sessions not yet started are skipped (OUT/skipped) for a
# later resume. With LEDGER (lines "KEY CREDITS", the last line of a key counting; it outlives the batch and is
# shared by every host and every resume of the evaluation), each session reserves RESERVE credits before it starts
# and settles at its measured credits (RESERVE when unmeasured) after; a session whose reservation would take the
# ledger over BUDGET is not started and creates STOP_FILE.
set -uo pipefail
# Episodes start together and share the candidate's folder on the group volume: concurrent bytecode-cache
# writes and renames there deadlocked a host's mount (virtiofs), so no episode writes __pycache__.
export PYTHONDONTWRITEBYTECODE=1
BASE=$1; PARALLEL=$2; JOBS=$3; EVOLVE=$4; MODEL=$5; SETTINGS=${6:-$4/episode_settings.toml}
STOP=${7:-}; BUDGET=${8:-0}; RESERVE=${9:-0}; LEDGER=${10:-}
W=$(dirname "$BASE")
QUOTA='usage limit|rate limit exceeded|insufficient_quota'
committed() {  # the ledger's credits: the last amount of every key
  python3 -c "import sys
last = {}
for line in open(sys.argv[1]):
    key, _, amount = line.rpartition(' ')
    if key:
        last[key] = float(amount)
print(round(sum(last.values()), 3))" "$LEDGER" 2>/dev/null || echo 0
}
reserve() {  # 0 when KEY's RESERVE fits under BUDGET (and is written); else STOP_FILE says why
  ( flock 9
    local total; total=$(committed)
    if python3 -c "import sys;sys.exit(0 if float(sys.argv[1]) + float(sys.argv[2]) <= float(sys.argv[3]) else 1)" \
        "$total" "$RESERVE" "$BUDGET"; then echo "$1 $RESERVE" >> "$LEDGER"
    else echo "the credit budget ($BUDGET) would be exceeded: $total committed" > "$STOP"; exit 1; fi
  ) 9>> "$LEDGER.lock"
}
settle() { ( flock 9; echo "$1 $2" >> "$LEDGER" ) 9>> "$LEDGER.lock"; }
run_one() {
  local cand=$1 scene=$2 targets=$3 out=$4
  mkdir -p "$out"
  if [ -n "$STOP" ] && [ -e "$STOP" ]; then echo "stopped: $(cat "$STOP")" > "$out/skipped"; return; fi
  local key="session:$out@$(date +%s%N)"
  if [ -n "$LEDGER" ] && ! reserve "$key"; then echo "stopped: $(cat "$STOP")" > "$out/skipped"; return; fi
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
  [ -n "$LEDGER" ] || return 0
  settle "$key" "$(python3 -c "import json,sys
u = json.load(open(sys.argv[1]))
print(u['credits'] if isinstance(u, dict) and not u.get('unmetered') and isinstance(u.get('credits'), (int, float)) else sys.argv[2])" \
    "$out/usage.json" "$RESERVE" 2>/dev/null || echo "$RESERVE")"
}
export -f run_one committed reserve settle
export BASE W EVOLVE MODEL SETTINGS STOP BUDGET RESERVE LEDGER QUOTA
xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$JOBS"
