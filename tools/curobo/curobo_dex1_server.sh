#!/usr/bin/env bash
# curobo_dex1_server.sh [start|stop|restart|status|check|watch] [COUNT]: this host's COUNT (default 3) cuRobo planning
# servers for the G1 with Dex1-1 grippers on tcp://127.0.0.1:5561.. (HomeBody's configs/dex1.toml [curobo] endpoint
# and servers), from the demo branch's tools/curobo_server.py copied here. Servers are found by their port, not a
# pid file. `start` waits until each answers and records the md5 of the server file it loaded
# (server-HOST-PORT.version); logs are appended, one "started" line per start. `check` prints one line per server,
# "PORT ready VERSION" only when it runs the current server file and passes its CUDA check (the server's "check"
# endpoint: CUDA work and each planner's kinematics), "PORT busy VERSION" when a session holds it (its admission
# lock, /tmp/homebody-curobo-PORT.lock) and its process is up, else "PORT down", "PORT silent", "PORT failed" or
# "PORT stale RECORDED"; it exits 1 unless all are ready or busy. Both inspect a server only while holding its
# admission lock themselves (a non-blocking try; a held server reads busy, or "PORT down held" when its process is
# gone), so no session can take it between the check and a restart. `watch` restarts, under that lock, every
# server that is down, stale or fails its check twice in a row (a server exits on a CUDA error,
# tools/curobo_server.py), and prints "PORT restarted REASON" for each, then the check lines.
W=/lambda/nfs/group-volume/workcell-sim/homebody-baseline
H=$(hostname -s); COUNT=${2:-3}
SERVER=$W/curobo_dex1/curobo_server.py
server() { pgrep -f "[c]urobo_dex1/curobo_server.py.*tcp://127.0.0.1:$1\$"; }
ask() {  # 0 when the server on port $1 answers endpoint $3 with ok within $2 seconds (retrying a silent one)
  (cd "$W/curobo" && .venv/bin/python - "$1" "$2" "$3" << 'EOF'
import sys, time, zmq, msgpack
port, wait, endpoint = sys.argv[1], float(sys.argv[2]), sys.argv[3]
deadline = time.monotonic() + wait
while True:
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 20000)
    socket.connect(f"tcp://127.0.0.1:{port}")
    try:
        socket.send(msgpack.packb({"endpoint": endpoint}))
        reply = msgpack.unpackb(socket.recv(), raw=False)
        sys.exit(0 if isinstance(reply, dict) and reply.get("ok") else 2)
    except zmq.Again:
        socket.close()
        if time.monotonic() > deadline:
            sys.exit(1)
        time.sleep(2)
EOF
  )
}
launch() {  # start port $1's server and wait for it
  local LOG=$W/curobo_dex1/server-$H-$1.log
  echo "=== started $(date -u +%FT%TZ) version $current" >> "$LOG"
  cd "$W/curobo" && setsid -f .venv/bin/python "$SERVER" --curobo-root "$W/curobo" \
    --hand "$W/homebody-astra/assets/robot/dex1" --endpoint tcp://127.0.0.1:$1 >> "$LOG" 2>&1 < /dev/null 9>&-
  echo "$current" > "$W/curobo_dex1/server-$H-$1.version"
  ask $1 180 ping
}
halt() { pids=$(server $1); [ -n "$pids" ] && kill $pids; for _ in $(seq 1 20); do [ -z "$(server $1)" ] && break; sleep 1; done
  [ -n "$(server $1)" ] && kill -9 $(server $1); true; }
state() {  # one check line for port $1, whose admission lock the caller holds
  local recorded
  recorded=$(cat "$W/curobo_dex1/server-$H-$1.version" 2>/dev/null || echo unknown)
  if [ -z "$(server $1)" ]; then echo "$1 down"
  elif [ "$recorded" != "$current" ]; then echo "$1 stale $recorded"
  else ask $1 10 check; case $? in 0) echo "$1 ready $current";; 1) echo "$1 silent";; *) echo "$1 failed";; esac
  fi
}
inspect() {  # port $1's check line, restarting it when $2 is watch, all under its admission lock (fd 9)
  local line again
  exec 9> "/tmp/homebody-curobo-$1.lock"
  if ! flock -n 9; then
    exec 9>&-
    if [ -n "$(server $1)" ]; then echo "$1 busy $current"; else echo "$1 down held"; fi
    return
  fi
  line=$(state $1)
  if [ "$2" = watch ] && ! [[ "$line" =~ \ ready\  ]]; then
    again=$(state $1)  # revalidate before halting: a slow answer once is not a broken server
    if ! [[ "$again" =~ \ ready\  ]]; then
      halt $1; launch $1 > /dev/null; echo "$1 restarted ${again#$1 }"; line=$(state $1)
    else line=$again
    fi
  fi
  flock -u 9; exec 9>&-
  echo "$line"
}
current=$(md5sum "$SERVER" | cut -c1-12)
status=0
for i in $(seq 0 $((COUNT - 1))); do
  PORT=$((5561 + i))
  case "${1:-start}" in
    start|restart)
      [ "${1:-start}" = restart ] && halt $PORT
      if [ -n "$(server $PORT)" ]; then echo "$PORT running $(server $PORT)"; continue; fi
      if launch $PORT; then echo "$PORT started $(server $PORT)"; else echo "$PORT started but silent"; status=1; fi ;;
    stop)
      halt $PORT; echo "$PORT stopped" ;;
    status)
      if [ -n "$(server $PORT)" ]; then echo "$PORT running $(server $PORT)"; else echo "$PORT down"; fi ;;
    check|watch)
      lines=$(inspect $PORT "$1"); echo "$lines"
      [[ "$(echo "$lines" | tail -1)" =~ \ (ready|busy)\  ]] || status=1 ;;
  esac
done
exit $status
