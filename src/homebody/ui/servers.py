"""The dashboard's System status in the monitor app: one row per harness and per simulation server, with its health
(healthy, degraded, unavailable, unknown), when it last answered, and what the running task depends on.

A server's task dependency comes only from established links, never from its name: the harness the dashboard drives
runs the task; a simulation server is that harness's when the harness itself reports being attached to its endpoint.
Without that report the dependency is unknown; a server nothing is attached to is not a dependency.
"""
import time

from homebody.server.watch import stale


def row(name, kind, health, latency, detail, error="", active=False, last_ok=None, dependency="", impact="unknown",
        connected=None, controller="unknown", video="unknown", checking=False):
    """CONNECTED: the server answered its last check (None: not checked yet, CHECKING); CONTROLLER (running, stopped,
    none, unknown) and VIDEO (on, ready, none, unknown) are a simulation server's readiness, apart from its health."""
    return {"name": name, "kind": kind, "health": health, "latency_ms": latency, "detail": detail, "error": error,
            "active": active, "last_ok": last_ok, "dependency": dependency, "impact": impact, "connected": connected,
            "controller": controller, "video": video, "checking": checking}


def harness_row(name, status, active, last_ok=None) -> dict:
    dependency, impact = ("Runs the task (the harness this dashboard drives)", "critical") if active \
        else ("Not used by this dashboard", "noncritical")
    extra = {"active": active, "last_ok": last_ok, "dependency": dependency, "impact": impact}
    if status is None:
        return row(name, "harness", "unknown", None, "not checked yet", checking=True, **extra)
    if not status.connected:
        return row(name, "harness", "unavailable", None, "", status.error or "no answer", connected=False, **extra)
    state = status.detail
    link = state.get("sim")
    if link is None:
        where, health = "physics in the harness", "healthy"
    elif link.get("connected"):
        where, health = f"on {link.get('profile', '?')} ({link.get('endpoint', '?')})", "healthy"
    else:
        where, health = f"lost its simulation server {link.get('endpoint', '?')}", "degraded"
    error = link.get("error", "") if link and not link.get("connected") else ""
    return row(name, "harness", health, status.latency_ms, f"{state.get('status', '?')} · {where}", error,
               connected=True, **extra)


def sim_dependency(sim, harness_status, active_sim):
    """(dependency, impact) of simulation server SIM, from what the harness in use reports it is attached to."""
    if harness_status is None or not harness_status.connected:
        if sim.name == active_sim:
            return "The last scenario's server; the harness's link to it is not confirmed", "unknown"
        return "No harness in use reports it", "noncritical"
    link = harness_status.detail.get("sim") or {}
    if link.get("endpoint") == sim.endpoint:
        return "The active harness runs on it", "critical"
    return "Not the active harness's server", "noncritical"


def sim_row(sim, status, active, harness_status=None, active_sim=None, last_ok=None) -> dict:
    dependency, impact = sim_dependency(sim, harness_status, active_sim)
    extra = {"active": active, "last_ok": last_ok, "dependency": dependency, "impact": impact}
    if status is None:
        return row(sim.name, "simulation", "unknown", None, "not checked yet", checking=True, **extra)
    if not status.connected:
        return row(sim.name, "simulation", "unavailable", None, "", status.error or "no answer", connected=False, **extra)
    detail = status.detail
    controller = detail.get("controller") or {}
    if not controller:
        control, health = "no controller plugin", "degraded"
    elif controller.get("running"):
        control, health = "controller running", "healthy"
    else:
        control, health = "controller stopped", "degraded"
    overshoot = detail.get("overshoot") or {}
    worst = max(overshoot.items(), key=lambda item: item[1], default=None)
    video = f"video {'on' if active else 'ready'}" if detail.get("stream") else "no stream"
    text = " · ".join([detail.get("profile", "?"), control, video]
                      + ([f"largest overshoot {worst[0]} {worst[1]:.3f}"] if worst else []))
    return row(sim.name, "simulation", health, status.latency_ms, text, controller.get("error") or "", connected=True,
               controller="none" if not controller else "running" if controller.get("running") else "stopped",
               video=("on" if active else "ready") if detail.get("stream") else "none", **extra)


def server_rows(config, statuses: dict, active_harness=None, active_sim=None, now=None, last_ok=None) -> list[dict]:
    now = time.time() if now is None else now
    last_ok = last_ok or {}
    harness_status = stale(statuses.get(("harness", active_harness)), now) if active_harness else None
    rows = [harness_row(name, stale(statuses.get(("harness", name)), now), name == active_harness,
                        last_ok.get(("harness", name))) for name in config.harnesses]
    return rows + [sim_row(sim, stale(statuses.get(("sim", name)), now), name == active_sim, harness_status,
                           active_sim, last_ok.get(("sim", name))) for name, sim in config.sims.items()]
