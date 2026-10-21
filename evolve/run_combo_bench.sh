#!/bin/bash
# run_combo_bench.sh NAME AGENT VENV LOCAL_REPO LOCAL_SERVED "LOCAL_ARGS" SECOND_REPO SECOND_SERVED "SECOND_ARGS":
# serve a multi-model agent's two models on this GPU host (127.0.0.1:8000 and :8001, the bigger first) and run the
# full-turn capability bench as AGENT (a session/config.py COMPOSITES name). Each ARGS string carries that server's
# vLLM flags, its --gpu-memory-utilization share included. TURN_ARGS adds turns flags (--layout images-first);
# CASES defaults to ~/bench/cases-all. Results go to ~/bench/runs/NAME-turns; both servers stop at the end.
set -u
NAME=$1 AGENT=$2 VENV=$3 LREPO=$4 LSERVED=$5 LARGS=$6 SREPO=$7 SSERVED=$8 SARGS=$9
JOBS=${JOBS:-16} CASES=${CASES:-$HOME/bench/cases-all} PY=${PY:-python3} RUNS=$HOME/bench/runs
STRUCTURED=${STRUCTURED:-'{"backend": "xgrammar", "disable_any_whitespace": true}'}
export HF_HOME=$HOME/hf VLLM_API_KEY
VLLM_API_KEY=$(cat ~/vlm/api_key)
mkdir -p "$RUNS"
cd "$(dirname "$0")/.." || exit 1

PIDS=()
serve() {  # REPO SERVED PORT "ARGS": start one server and wait until it answers
  # shellcheck disable=SC2086 # ARGS is a flag list
  setsid nohup "$VENV/bin/vllm" serve "$1" --served-model-name "$2" --host 127.0.0.1 --port "$3" \
    --max-model-len 16384 --limit-mm-per-prompt '{"image": 4}' --enable-prefix-caching \
    --structured-outputs-config "$STRUCTURED" $4 > "$RUNS/$NAME-$3-server.log" 2>&1 < /dev/null &
  PIDS+=($!)
  for _ in $(seq 360); do
    curl -sf -H "Authorization: Bearer $VLLM_API_KEY" "http://127.0.0.1:$3/v1/models" | grep -q "$2" && return 0
    kill -0 "${PIDS[-1]}" 2>/dev/null || break
    sleep 10
  done
  echo "$NAME: $2 did not start; see $RUNS/$NAME-$3-server.log"
  return 1
}
stop() { for pid in "${PIDS[@]}"; do kill -- -"$pid" 2>/dev/null; done; sleep 10; }

if serve "$LREPO" "$LSERVED" 8000 "$LARGS" && serve "$SREPO" "$SSERVED" 8001 "$SARGS"; then
  # shellcheck disable=SC2086 # TURN_ARGS: extra agent_bench turns flags
  PYTHONPATH=src:. "$PY" -m evolve.agent_bench turns "$CASES" "$RUNS/$NAME-turns" --model "$AGENT" \
    --local-url http://127.0.0.1:8000/v1 --second-url http://127.0.0.1:8001/v1 --label "$NAME" --jobs "$JOBS" \
    ${TURN_ARGS:-} > "$RUNS/$NAME-turns.log" 2>&1
  echo "$NAME: done"
fi
stop
