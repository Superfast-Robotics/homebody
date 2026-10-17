"""The recorded settings mask the robot password without changing the live settings."""
import json
import threading
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

from homebody.real2sim.exploration import settings
from homebody.real2sim.exploration.explorer import Explorer
from homebody.real2sim.exploration.livemap import Boundary
from homebody.session import exploration as session_module
from homebody.session.live_recorder import LiveRecorder


def test_session_provenance_masks_password_and_keeps_runtime_settings(tmp_path, monkeypatch):
    sentinel = "synthetic-test-password-never-use-for-authentication"
    defaults = settings.Settings()
    configured = replace(
        defaults,
        robot=replace(defaults.robot, ssh_password=sentinel, iface="synthetic-interface"),
        exploration=replace(defaults.exploration, task="Configured task", budget_s=210.),
        drive=replace(defaults.drive, tolerance_m=.375),
    )
    monkeypatch.setitem(settings._installed, "settings", configured)
    monkeypatch.delenv("HOMEBODY_G1_SSH_PASSWORD", raising=False)
    provider = Mock(side_effect=AssertionError("Constructing a session must not call a provider"))
    monkeypatch.setattr(session_module.vlm_pick, "make", provider)
    source = SimpleNamespace(stop_flag=threading.Event(), pause_flag=threading.Event(),
                             attach=lambda *_: None)
    boundary = Boundary(((-2., -2.), (4., -2.), (4., 4.), (-2., 4.)))
    explorer = Explorer(source, boundary, tmp_path, budget_s=83.)
    explorer.brief = "Inspect the synthetic room"

    session = session_module.ExplorationSession(explorer)
    try:
        assert isinstance(session.recorder, LiveRecorder)
        assert session.settings is configured
        assert session.settings.robot.ssh_password == sentinel
        assert settings.ssh_password(session.settings) == sentinel
    finally:
        session.recorder.close()

    provenance_text = (tmp_path / "session" / "provenance.json").read_text()
    assert sentinel not in provenance_text
    provenance = json.loads(provenance_text)
    recorded = provenance["settings"]
    assert recorded["robot"]["ssh_password"] == settings.PASSWORD_MASK
    assert recorded["robot"]["iface"] == "synthetic-interface"
    assert recorded["robot"]["ssh_target"] == configured.robot.ssh_target
    assert recorded["exploration"] == asdict(configured.exploration)
    assert recorded["drive"] == asdict(configured.drive)
    assert recorded["agent"] == asdict(configured.agent)
    assert recorded["task"] == explorer.brief
    assert recorded["budget_s"] == 83.
    assert provenance["backend"] == "g1" and provenance["schema_version"] == 1
    assert json.loads((tmp_path / "session" / "finalization.json").read_text())["status"] == "closed"
    provider.assert_not_called()
