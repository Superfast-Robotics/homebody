#!/bin/bash
# run_bench.sh NAME REPO SERVED POINT_FORMAT AGENT_MODEL VENV [vLLM ARGS...]: serve one model on this GPU host
# (127.0.0.1:$PORT) and run the capability bench on it (evolve/agent_bench.py): full decision turns as AGENT_MODEL
# (a session/config.py MODELS name, "-" for a pointer that is no agent), pointing in POINT_FORMAT, yes/no checks.
# Cases are ~/bench/cases-all (every recorded turn); results go to ~/bench/runs/NAME-*. The server stops at the end.
# SUITES (default "turns point verify") picks the benches, TURN_ARGS adds turns flags; a run resumes where it stopped.
set -u
NAME=$1 REPO=$2 SERVED=$3 FORMAT=$4 AGENT=$5 VENV=$6
shift 6
PORT=${PORT:-8000} JOBS=${JOBS:-16} SUITES=${SUITES:-turns point verify} CASES=${CASES:-$HOME/bench/cases-all} PY=${PY:-python3}
URL=http://127.0.0.1:$PORT/v1 RUNS=$HOME/bench/runs
export VLLM_API_KEY HF_HOME=$HOME/hf
VLLM_API_KEY=$(cat ~/vlm/api_key)
mkdir -p "$RUNS"
cd "$(dirname "$0")/.." || exit 1

setsid nohup "$VENV/bin/vllm" serve "$REPO" --served-model-name "$SERVED" --host 127.0.0.1 --port "$PORT" \
  --max-model-len 16384 --limit-mm-per-prompt '{"image": 4}' --enable-prefix-caching "$@" \
  > "$RUNS/$NAME-server.log" 2>&1 < /dev/null &
for _ in $(seq 360); do
  curl -sf -H "Authorization: Bearer $VLLM_API_KEY" "$URL/models" | grep -q "$SERVED" && break
  sleep 10
done
curl -sf -H "Authorization: Bearer $VLLM_API_KEY" "$URL/models" | grep -q "$SERVED" || {
  echo "$NAME: the server did not start; see $RUNS/$NAME-server.log"; pkill -f "vllm serve .*--port $PORT( |$)"; exit 1; }

bench() { PYTHONPATH=src:. "$PY" -m evolve.agent_bench "$@" --jobs "$JOBS"; }
if [ "$AGENT" != "-" ] && [[ " $SUITES " == *" turns "* ]]; then
  # shellcheck disable=SC2086 # TURN_ARGS: extra agent_bench turns flags (--layout, --no-labels)
  bench turns "$CASES" "$RUNS/$NAME-turns" --model "$AGENT" --local-url "$URL" --label "$NAME" ${TURN_ARGS:-} \
    > "$RUNS/$NAME-turns.log" 2>&1
fi
[[ " $SUITES " == *" point "* ]] && bench point "$CASES" "$RUNS/$NAME-point" --served "$SERVED" --format "$FORMAT" --url "$URL" --label "$NAME" \
  > "$RUNS/$NAME-point.log" 2>&1
[[ " $SUITES " == *" verify "* ]] && bench verify "$CASES" "$RUNS/$NAME-verify" --served "$SERVED" --url "$URL" --label "$NAME" > "$RUNS/$NAME-verify.log" 2>&1
pkill -f "vllm serve .*--port $PORT( |$)"
echo "$NAME: done"
