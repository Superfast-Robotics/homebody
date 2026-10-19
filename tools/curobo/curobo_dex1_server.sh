#!/usr/bin/env bash
# curobo_dex1_server.sh [start|stop|status] [COUNT]: this host's COUNT (default 3) cuRobo planning servers for the
# G1 with Dex1-1 grippers on tcp://127.0.0.1:5561.. (HomeBody's configs/dex1.toml [curobo] endpoint and servers),
# from the demo branch's tools/curobo_server.py copied here. Servers are found by their port, not a pid file.
W=/lambda/nfs/group-volume/workcell-sim/homebody-baseline
H=$(hostname -s); COUNT=${2:-3}
server() { pgrep -f "[c]urobo_dex1/curobo_server.py.*tcp://127.0.0.1:$1\$"; }
for i in $(seq 0 $((COUNT - 1))); do
  PORT=$((5561 + i)); LOG=$W/curobo_dex1/server-$H-$PORT.log
  case "${1:-start}" in
    start)
      if [ -n "$(server $PORT)" ]; then echo "$PORT running $(server $PORT)"; continue; fi
      cd "$W/curobo" && setsid -f .venv/bin/python "$W/curobo_dex1/curobo_server.py" --curobo-root "$W/curobo" \
        --hand "$W/homebody-astra/assets/robot/dex1" --endpoint tcp://127.0.0.1:$PORT > "$LOG" 2>&1 < /dev/null
      sleep 1; echo "$PORT started $(server $PORT)" ;;
    stop)
      pids=$(server $PORT); [ -n "$pids" ] && kill $pids; for _ in 1 2 3 4 5 6 7 8 9 10; do [ -z "$(server $PORT)" ] && break; sleep 1; done
      echo "$PORT stopped" ;;
    status)
      if [ -n "$(server $PORT)" ]; then echo "$PORT running $(server $PORT)"; else echo "$PORT down"; fi ;;
  esac
done
