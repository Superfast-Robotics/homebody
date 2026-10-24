"""The dashboard's Servers list in the monitor app: one row per harness and per simulation server (is it answering,
how fast, what it runs), each harness's own link to its simulation server. The page draws the rows."""
import time

from homebody.server.watch import stale

UP, DOWN, WARN = "up", "down", "unknown"


def row(name, kind, health, latency, detail, error="", active=False) -> dict:
    return {"name": name, "kind": kind, "health": health, "latency_ms": latency, "detail": detail, "error": error,
            "active": active}


def harness_row(name, status, active) -> dict:
    if status is None:
        return row(name, "harness", WARN, None, "not checked yet", active=active)
    if not status.connected:
        return row(name, "harness", DOWN, None, "", status.error or "no answer", active)
    state = status.detail
    link = state.get("sim")
    if link is None:
        where, health = "physics in the harness", UP
    elif link.get("connected"):
        where, health = f"on {link.get('profile', '?')} ({link.get('endpoint', '?')})", UP
    else:
        where, health = f"lost its simulation server {link.get('endpoint', '?')}", DOWN
    error = link.get("error", "") if link and not link.get("connected") else ""
    return row(name, "harness", health, status.latency_ms, f"{state.get('status', '?')} · {where}", error, active)


def sim_row(name, status, active) -> dict:
    if status is None:
        return row(name, "simulation", WARN, None, "not checked yet", active=active)
    if not status.connected:
        return row(name, "simulation", DOWN, None, "", status.error or "no answer", active)
    detail = status.detail
    controller = detail.get("controller") or {}
    if not controller:
        control, health = "no controller plugin", WARN
    elif controller.get("running"):
        control, health = "controller running", UP
    else:
        control, health = "controller stopped", DOWN
    overshoot = detail.get("overshoot") or {}
    worst = max(overshoot.items(), key=lambda item: item[1], default=None)
    video = f"video {'on' if active else 'ready'}" if detail.get("stream") else "no stream"
    text = " · ".join([detail.get("profile", "?"), control, video]
                      + ([f"largest overshoot {worst[0]} {worst[1]:.3f}"] if worst else []))
    return row(name, "simulation", health, status.latency_ms, text, controller.get("error") or "", active)


def server_rows(config, statuses: dict, active_harness=None, active_sim=None, now=None) -> list[dict]:
    now = time.time() if now is None else now
    rows = [harness_row(name, stale(statuses.get(("harness", name)), now), name == active_harness)
            for name in config.harnesses]
    return rows + [sim_row(name, stale(statuses.get(("sim", name)), now), name == active_sim) for name in config.sims]
