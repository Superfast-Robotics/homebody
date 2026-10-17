"""Open both Dex3 hands with a slow ramp through the G1 Dex3 transport. Hand topic only.

    python hardware/g1/open_hands.py --settings-json RUN/meta/settings.json

Writes rt/dex3/<side>/cmd only — never a body step, never rt/lowcmd — at the transport's
own rate limit (HAND_MAX_RATE). The exploration launcher invokes it before the console starts.
LOOK at the hands first: an open ramp drops whatever they hold. Stops when both hands read
under 15% closed (--floor) or after --seconds. Watch the hands while it runs; Ctrl-C stops
it at once.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=25.0, help="ramp budget")
    ap.add_argument("--floor", type=float, default=0.15, help="closure that counts as open")
    ap.add_argument("--settings-json", type=Path, required=True,
                    help="the run's resolved settings: the robot interface and the hand gains")
    args = ap.parse_args()

    from homebody.real2sim.exploration.settings import current, from_json, install

    install(from_json(args.settings_json.read_text()))

    from homebody.backends.g1 import profile as g1
    from homebody.backends.g1.dex3 import Dex3Port

    iface = current().robot.iface
    if not iface:
        raise SystemExit("the settings name no robot interface (robot.iface)")
    port = Dex3Port(iface)
    open_q = np.asarray(g1.HAND_OPEN, dtype=float)
    state = port.state()
    print(f"before: L={state['closure_l']:.2f} R={state['closure_r']:.2f} closed", flush=True)
    port.set_targets(open_q, open_q)
    t0 = last = time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            now = time.monotonic()
            port.tick(now - last)
            last = now
            state = port.state()
            if int((now - t0) * 5) % 5 == 0:
                print(f"  t={now - t0:4.1f}s  L={state['closure_l']:.2f} "
                      f"R={state['closure_r']:.2f}", flush=True)
            if state["closure_l"] < args.floor and state["closure_r"] < args.floor:
                print(f"open: L={state['closure_l']:.2f} R={state['closure_r']:.2f}", flush=True)
                break
            time.sleep(0.02)
        else:
            print(f"still not under {args.floor:.0%} after {args.seconds:.0f}s: "
                  f"L={state['closure_l']:.2f} R={state['closure_r']:.2f} — "
                  f"unpowered, obstructed, or holding something", flush=True)
            return 1
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        port.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
