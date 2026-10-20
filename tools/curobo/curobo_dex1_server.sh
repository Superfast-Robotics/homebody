#!/usr/bin/env bash
# curobo_dex1_server.sh [start|stop|restart|status|check] [COUNT]: this host's COUNT (default 3) cuRobo planning
# servers for the G1 with Dex1-1 grippers on tcp://127.0.0.1:5561.. (HomeBody's configs/dex1.toml [curobo] endpoint
# and servers), from the demo branch's tools/curobo_server.py copied here. Servers are found by their port, not a
# pid file. `start` waits until each answers a ping and records the md5 of the server file it loaded
# (server-HOST-PORT.version); `check` prints one line per server, "PORT ready VERSION" only when it answers and runs
# the current server file ("PORT down", "PORT silent" or "PORT stale RECORDED" otherwise), and exits 1 unless all
# are ready. A server started before versions were recorded reads as stale: restart it.
W=/lambda/nfs/group-volume/workcell-sim/homebody-baseline
H=$(hostname -s); COUNT=${2:-3}
SERVER=$W/curobo_dex1/curobo_server.py
server() { pgrep -f "[c]urobo_dex1/curobo_server.py.*tcp://127.0.0.1:$1\$"; }
ping() {  # 0 when the server on port $1 answers within $2 seconds
  (cd "$W/curobo" && .venv/bin/python - "$1" "$2" << 'EOF'
import sys, time, zmq, msgpack
port, wait = sys.argv[1], float(sys.argv[2])
deadline = time.monotonic() + wait
while True:
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.connect(f"tcp://127.0.0.1:{port}")
    try:
        socket.send(msgpack.packb({"endpoint": "ping"}))
        socket.recv()
        sys.exit(0)
    except zmq.Again:
        socket.close()
        if time.monotonic() > deadline:
            sys.exit(1)
        time.sleep(2)
EOF
  )
}
current=$(md5sum "$SERVER" | cut -c1-12)
status=0
for i in $(seq 0 $((COUNT - 1))); do
  PORT=$((5561 + i)); LOG=$W/curobo_dex1/server-$H-$PORT.log; VERSION=$W/curobo_dex1/server-$H-$PORT.version
  case "${1:-start}" in
    start|restart)
      if [ "${1:-start}" = restart ] && [ -n "$(server $PORT)" ]; then
        kill $(server $PORT); for _ in $(seq 1 20); do [ -z "$(server $PORT)" ] && break; sleep 1; done
      fi
      if [ -n "$(server $PORT)" ]; then echo "$PORT running $(server $PORT)"; continue; fi
      cd "$W/curobo" && setsid -f .venv/bin/python "$SERVER" --curobo-root "$W/curobo" \
        --hand "$W/homebody-astra/assets/robot/dex1" --endpoint tcp://127.0.0.1:$PORT > "$LOG" 2>&1 < /dev/null
      echo "$current" > "$VERSION"
      if ping $PORT 180; then echo "$PORT started $(server $PORT)"; else echo "$PORT started but silent"; status=1; fi ;;
    stop)
      pids=$(server $PORT); [ -n "$pids" ] && kill $pids; for _ in 1 2 3 4 5 6 7 8 9 10; do [ -z "$(server $PORT)" ] && break; sleep 1; done
      echo "$PORT stopped" ;;
    status)
      if [ -n "$(server $PORT)" ]; then echo "$PORT running $(server $PORT)"; else echo "$PORT down"; fi ;;
    check)
      recorded=$(cat "$VERSION" 2>/dev/null || echo unknown)
      if [ -z "$(server $PORT)" ]; then echo "$PORT down"; status=1
      elif [ "$recorded" != "$current" ]; then echo "$PORT stale $recorded"; status=1
      elif ! ping $PORT 10; then echo "$PORT silent"; status=1
      else echo "$PORT ready $current"; fi ;;
  esac
done
exit $status
