"""The console's Servers section in the monitor app: one row per harness and per simulation server (is it answering,
how fast, what it runs), each harness's own link to its simulation server, and Refresh config."""
import time
from html import escape

from homebody.server.watch import stale

STYLE = """<style>
.sv-rows{font:13px/1.35 var(--op-sans,'IBM Plex Sans',sans-serif);color:var(--op-ink,#14202b)}
.sv-row{display:grid;grid-template-columns:12px 1fr auto;gap:8px;align-items:baseline;padding:5px 0;
  border-bottom:1px solid var(--op-rule,#dfe3e7)}
.sv-dot{width:9px;height:9px;border-radius:50%;display:inline-block;transform:translateY(1px)}
.sv-name{font-weight:600}.sv-kind{color:var(--op-muted,#5e6a75);font-weight:400}
.sv-detail{color:var(--op-muted,#5e6a75);font-size:12px;grid-column:2/4}
.sv-ms{color:var(--op-muted,#5e6a75);font-variant-numeric:tabular-nums}
.sv-bad{color:#c2271d}
</style>"""
UP, DOWN, WARN = "#2e7a57", "#c2271d", "#d89a00"


def row(name, kind, colour, latency, detail, error="") -> str:
    ms = f"{latency:.0f} ms" if latency is not None else "—"
    text = escape(detail) + (f' <span class="sv-bad">{escape(error)}</span>' if error else "")
    return (f'<div class="sv-row"><i class="sv-dot" style="background:{colour}"></i>'
            f'<span class="sv-name">{escape(name)} <span class="sv-kind">{kind}</span></span>'
            f'<span class="sv-ms">{ms}</span><span class="sv-detail">{text}</span></div>')


def harness_row(name, status, active) -> str:
    if status is None:
        return row(name, "harness", WARN, None, "not checked yet")
    if not status.connected:
        return row(name, "harness", DOWN, None, "", status.error or "no answer")
    state = status.detail
    link = state.get("sim")
    if link is None:
        where, colour = "physics in the harness", UP
    elif link.get("connected"):
        where, colour = f"on {link.get('profile', '?')} ({link.get('endpoint', '?')})", UP
    else:
        where, colour = f"lost its simulation server {link.get('endpoint', '?')}", DOWN
    detail = f"{state.get('status', '?')} · {where}" + (" · this console" if active else "")
    return row(name, "harness", colour, status.latency_ms, detail, link.get("error", "") if link and not link.get("connected") else "")


def sim_row(name, status, active) -> str:
    if status is None:
        return row(name, "simulation", WARN, None, "not checked yet")
    if not status.connected:
        return row(name, "simulation", DOWN, None, "", status.error or "no answer")
    detail = status.detail
    controller = detail.get("controller") or {}
    if not controller:
        control, colour = "no controller plugin", WARN
    elif controller.get("running"):
        control, colour = "controller running", UP
    else:
        control, colour = "controller stopped", DOWN
    overshoot = detail.get("overshoot") or {}
    worst = max(overshoot.items(), key=lambda item: item[1], default=None)
    text = f"{detail.get('profile', '?')} · {control}" + (f" · video {'on' if active else 'ready'}" if detail.get("stream") else
                                                          " · no stream") + \
        (f" · largest overshoot {worst[0]} {worst[1]:.3f}" if worst else "")
    return row(name, "simulation", colour, status.latency_ms, text, controller.get("error") or "")


def servers_html(config, statuses: dict, active_harness=None, active_sim=None, now=None) -> str:
    now = time.time() if now is None else now
    rows = [harness_row(name, stale(statuses.get(("harness", name)), now), name == active_harness)
            for name in config.harnesses]
    rows += [sim_row(name, stale(statuses.get(("sim", name)), now), name == active_sim) for name in config.sims]
    return '<div class="sv-rows">' + "".join(rows) + "</div>"


class ServersPanel:
    """The Servers section's widgets, redrawn from the watcher's statuses."""

    def __init__(self, app):
        from homebody.ui.theme import section
        self.app = app
        gui = app.console.server.gui
        gui.add_html(section("Servers") + STYLE)
        self.rows = gui.add_html(servers_html(app.config, {}))
        self.refresh_button = gui.add_button("Refresh config", icon="refresh", color="gray",
                                             hint=f"Read {app.config_path.name} again")
        self.message = gui.add_markdown("")
        self.refresh_button.on_click(lambda _: self.refresh())

    def draw(self, statuses):
        self.rows.content = servers_html(self.app.config, statuses, self.app.harness,
                                         self.app.sim.name if self.app.sim else None)

    def refresh(self):
        self.message.content = escape(self.app.refresh_config())
        self.draw(self.app.watcher.refresh())
