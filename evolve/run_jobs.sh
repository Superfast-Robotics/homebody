#!/usr/bin/env bash
# run_jobs.sh BASE PARALLEL JOBS_FILE EVOLVE_DIR MODEL [SETTINGS [STOP_FILE [BUDGET RESERVE LEDGER [SANDBOX]]]]: run every
# "CANDIDATE SCENE TARGETS OUT" line of JOBS_FILE as one HomeBody session (the agent may customize skills) with the
# candidate's code, then score it from physics with the loop's own evaluator (EVOLVE_DIR/evaluate.py, never the
# candidate's copy). The agent model is reached through the LiteLLM proxy tunnel (port 14000) and the coding
# sub-agent through the Codex relay.
# Each OUT gets usage.json (the session's credits, whether or not it scores), result.json (only when scoring
# succeeded; the scorer's own output stays in score.out and score.err) and exit_code. With STOP_FILE, a session
# that hit the model's usage limit creates STOP_FILE: the sessions not yet started are skipped (OUT/skipped) for a
# later resume. With LEDGER (lines "KEY CREDITS", the last line of a key counting; it outlives the batch and is
# shared by every host, every resume and the round's Astra calls), each session reserves RESERVE credits (the worst
# case a session can cost) as "reserve:KEY" before it starts, and on finishing settles: the reservation goes to 0 and
# "KEY" to its measured credits (RESERVE when unmeasured), so the next session sees the actual spend at once. A
# reservation that would take the ledger over BUDGET waits while other reservations are in flight (they settle below
# their worst case) and, when none is, is refused: the session is not started and creates STOP_FILE.
# With SANDBOX=1 each session runs in bubblewrap with the whole filesystem read-only except its OUT folder, its
# candidate's walking-policy cache (CANDIDATE/.cache), /tmp (the planners' admission locks) and the coding sub-agent's
# home (~/.homebody-coder): candidate code cannot touch the scorer, settings, scenes, the seed or other sessions.
# Scoring runs outside, from EVOLVE_DIR.
set -uo pipefail
# Episodes start together and share the candidate's folder on the group volume: concurrent bytecode-cache
# writes and renames there deadlocked a host's mount (virtiofs), so no episode writes __pycache__.
export PYTHONDONTWRITEBYTECODE=1
BASE=$1; PARALLEL=$2; JOBS=$3; EVOLVE=$4; MODEL=$5; SETTINGS=${6:-$4/episode_settings.toml}
STOP=${7:-}; BUDGET=${8:-0}; RESERVE=${9:-0}; LEDGER=${10:-}; SANDBOX=${11:-0}
W=$(dirname "$BASE")
QUOTA='usage limit|rate limit exceeded|insufficient_quota'
LEDGER_PY='import sys
last = {}
for line in open(sys.argv[1]):
    key, _, amount = line.strip().rpartition(" ")
    try:
        if key:
            last[key] = float(amount)
    except ValueError:
        continue
total = sum(last.values())
flight = sum(v for k, v in last.items() if k.startswith("reserve:"))
reserve, budget = float(sys.argv[2]), float(sys.argv[3])
print(f"{total:.3f}")
sys.exit(0 if total + reserve <= budget else 2 if flight > 0 else 1)'
reserve() {  # 0 once KEY's RESERVE fits under BUDGET (written as reserve:KEY); 1 (STOP_FILE says why) when it cannot
  local rc total
  while true; do
    [ -e "$STOP" ] && return 1
    total=$( ( flock 9
      python3 -c "$LEDGER_PY" "$LEDGER" "$RESERVE" "$BUDGET"; rc=$?
      [ $rc = 0 ] && echo "reserve:$1 $RESERVE" >> "$LEDGER"; exit $rc ) 9>> "$LEDGER.lock")
    rc=$?
    [ $rc = 0 ] && return 0
    if [ $rc = 1 ]; then echo "the credit budget ($BUDGET) would be exceeded: $total committed" > "$STOP"; return 1; fi
    sleep 20  # others are in flight: their settlements free room
  done
}
jailed() {  # jailed DIR... -- COMMAND: COMMAND, in bubblewrap with only the DIRs writable when SANDBOX=1
  local binds=()
  while [ "$1" != -- ]; do mkdir -p "$1"; binds+=(--bind "$1" "$1"); shift; done; shift
  [ "$SANDBOX" = 1 ] || { "$@"; return; }
  mkdir -p ~/.homebody-coder
  bwrap --die-with-parent --unshare-pid --unshare-ipc --ro-bind / / --dev-bind /dev /dev --proc /proc --bind /tmp /tmp \
    "${binds[@]}" --bind ~/.homebody-coder ~/.homebody-coder -- "$@"
}
settle() { ( flock 9; printf 'reserve:%s 0\n%s %s\n' "$1" "$1" "$2" >> "$LEDGER" ) 9>> "$LEDGER.lock"; }
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
    PYTHONPATH="$cand/src:$cand/vendor" jailed "$out" "$cand/.cache" -- timeout 7200 "$BASE/.venv/bin/python" -m homebody --headless --root "$cand" \
    --scene "$scene" --model "$MODEL" --settings "$SETTINGS" --output "$out/session" \
    --task "$task" > "$out/run.log" 2>&1
  echo "$?" > "$out/exit_code"
  "$BASE/.venv/bin/python" "$EVOLVE/score_episode.py" --usage "$out/session" "$MODEL" > "$out/usage.json" 2>> "$out/score.err" \
    || echo '{"credits": null, "unmetered": true, "failed": true}' > "$out/usage.json"
  rm -f "$out/result.json"
  "$BASE/.venv/bin/python" "$EVOLVE/score_episode.py" "$out/session" "$targets" "$EVOLVE/evaluate.py" "$MODEL" \
    > "$out/score.out" 2>> "$out/score.err" && cp "$out/score.out" "$out/result.json"
  [ -n "$STOP" ] || return 0
  if grep -qiE "$QUOTA" "$out/run.log" "$out/session"/*/tasks/*/outcome.json 2>/dev/null; then
    echo "the model's usage limit was hit in $out" > "$STOP"
  fi
  [ -n "$LEDGER" ] || return 0
  settle "$key" "$(python3 -c "import json, math, sys
u = json.load(open(sys.argv[1]))
c = u.get('credits') if isinstance(u, dict) and not u.get('unmetered') else None
ok = isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) and c >= 0
print(c if ok else sys.argv[2])" "$out/usage.json" "$RESERVE" 2>/dev/null || echo "$RESERVE")"
}
export -f run_one reserve settle jailed
export BASE W EVOLVE MODEL SETTINGS STOP BUDGET RESERVE LEDGER QUOTA SANDBOX LEDGER_PY
xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$JOBS"
