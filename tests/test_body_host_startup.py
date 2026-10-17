"""Startup reserves only ephemeral loopback sockets; hardware and process creation are faked."""
import errno
import socket
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from homebody.backends.g1 import body_host, writer

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the sole-writer singleton binds a Linux abstract unix socket")


class StartupFailure(RuntimeError):
    pass


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(body_host, "HOST_LEASE", tmp_path / "host.lock")
    monkeypatch.setattr(body_host, "PIDFILE", tmp_path / "host.pid")
    monkeypatch.setattr(body_host, "LOGFILE", tmp_path / "host.log")
    monkeypatch.setattr(body_host, "lease_path", lambda: tmp_path / "writer.lock")
    monkeypatch.setattr(body_host, "_realtime", lambda _cores: "isolated test")
    monkeypatch.setattr(body_host, "validate_assets", Mock(return_value={}))
    monkeypatch.setattr(body_host, "signal", SimpleNamespace(
        SIGTERM=15, SIGHUP=1, SIGINT=2, SIGUSR1=10, signal=Mock()))
    monkeypatch.setattr(body_host, "gc", SimpleNamespace(collect=Mock(), freeze=Mock()))
    monkeypatch.setattr(body_host, "BodyClient", Mock(side_effect=AssertionError("client contact")))
    monkeypatch.setattr(body_host.subprocess, "run", Mock(side_effect=AssertionError("spawn")))
    monkeypatch.setattr(body_host.subprocess, "Popen", Mock(side_effect=AssertionError("spawn")))
    # Restoring the whole environment also contains serve's CPU-runtime environment changes.
    monkeypatch.setattr(body_host.os, "environ", dict(body_host.os.environ))
    def exit_process(code):
        raise SystemExit(code)

    monkeypatch.setattr(body_host.os, "_exit", Mock(side_effect=exit_process))
    port = Mock(side_effect=StartupFailure("fake Unitree constructor reached"))
    wireless = Mock(return_value=object())
    monkeypatch.setattr(body_host, "UnitreeG1Port", port)
    monkeypatch.setattr(body_host, "UnitreeWirelessInput", wireless)
    monkeypatch.setattr(body_host, "g1", SimpleNamespace(PROFILE=object()))
    return SimpleNamespace(port=port, wireless=wireless)


@pytest.fixture
def endpoint():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as owner:
        owner.bind(("127.0.0.1", 0))
        yield owner, f"127.0.0.1:{owner.getsockname()[1]}"


def args(bind):
    return SimpleNamespace(bind=bind, log="", cores="", iface="fake-interface", simulated=False,
                           use_sudo=False, no_systemd=True, settings_json="")


def assert_reserved(address):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as contender:
        with pytest.raises(OSError) as failure:
            contender.bind(body_host.parse_addr(address))
        assert failure.value.errno == errno.EADDRINUSE


def assert_released(address):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as contender:
        contender.bind(body_host.parse_addr(address))
    assert not body_host.lease_held(body_host.HOST_LEASE)


def test_serve_refuses_occupied_endpoint_before_unitree(isolated, endpoint):
    owner, address = endpoint
    with pytest.raises(SystemExit) as failure:
        body_host.serve(args(address))
    assert address in str(failure.value)
    isolated.port.assert_not_called()
    isolated.wireless.assert_not_called()
    assert owner.fileno() >= 0
    assert_reserved(address)
    assert not body_host.lease_held(body_host.HOST_LEASE)


def test_start_refuses_occupied_endpoint_without_spawning(isolated, endpoint, capsys):
    _owner, address = endpoint
    assert body_host.start(args(address)) == 1
    assert address in capsys.readouterr().err
    body_host.subprocess.run.assert_not_called()
    body_host.subprocess.Popen.assert_not_called()
    isolated.port.assert_not_called()
    assert_reserved(address)


def test_start_reuses_recognized_host_without_spawning(isolated, endpoint, monkeypatch):
    _owner, address = endpoint
    monkeypatch.setattr(body_host, "_stale_host", lambda _simulated: "")
    monkeypatch.setattr(body_host, "running_pid", lambda: 123)
    lease = writer.SoleWriter(body_host.HOST_LEASE)
    try:
        assert body_host.start(args(address)) == 0
        body_host.subprocess.run.assert_not_called()
        body_host.subprocess.Popen.assert_not_called()
        isolated.port.assert_not_called()
        assert_reserved(address)
    finally:
        lease.release()


def test_serve_reserves_before_unitree_and_cleans_failed_constructor(isolated, endpoint):
    owner, address = endpoint
    owner.close()

    def fail_constructor(*_args):
        assert_reserved(address)
        assert body_host.lease_held(body_host.HOST_LEASE)
        raise StartupFailure("injected constructor failure")

    isolated.port.side_effect = fail_constructor
    with pytest.raises(StartupFailure, match="injected constructor failure"):
        body_host.serve(args(address))
    assert_released(address)
    assert not body_host.PIDFILE.exists()


def test_serve_cleans_reservation_and_lease_if_host_construction_fails(
        isolated, endpoint, monkeypatch):
    owner, address = endpoint
    owner.close()
    isolated.port.side_effect = None
    isolated.port.return_value = object()
    captured = []

    def fail_host(_port, *, sock, **_kwargs):
        captured.append(sock)
        assert not sock.getblocking()
        assert_reserved(address)
        raise StartupFailure("injected host failure")

    monkeypatch.setattr(body_host, "BodyHost", fail_host)
    with pytest.raises(StartupFailure, match="injected host failure"):
        body_host.serve(args(address))
    assert captured[0].fileno() == -1
    assert_released(address)
    assert not body_host.PIDFILE.exists()


@pytest.mark.parametrize("run_fails", [False, True])
def test_serve_keeps_reservation_until_run_ends(isolated, endpoint, monkeypatch, run_fails):
    owner, address = endpoint
    owner.close()
    isolated.port.side_effect = None
    isolated.port.return_value = object()
    captured = []

    def make_host(_port, *, sock, **_kwargs):
        captured.append(sock)
        assert_reserved(address)

        def run():
            assert_reserved(address)
            assert body_host.lease_held(body_host.HOST_LEASE)
            assert body_host.PIDFILE.exists()
            if run_fails:
                raise StartupFailure("injected run failure")

        return SimpleNamespace(sock=sock, run=run, event=Mock())

    monkeypatch.setattr(body_host, "BodyHost", make_host)
    with pytest.raises(SystemExit) as exit_:
        body_host.serve(args(address))
    assert exit_.value.code == int(run_fails)
    assert captured[0].fileno() == -1
    assert_released(address)
    assert not body_host.PIDFILE.exists()


@pytest.mark.parametrize("prebound", [False, True])
def test_direct_body_host_uses_owned_socket(isolated, monkeypatch, prebound):
    profile = body_host.g1
    profile.ARM_SLICE = {"left": slice(15, 22), "right": slice(22, 29)}
    profile.ArmGate = Mock(side_effect=lambda **_kwargs: Mock())
    monkeypatch.setattr(body_host, "load_policy", Mock(return_value=object()))
    monkeypatch.setattr(body_host, "_limit_cpu_threads", Mock())
    monkeypatch.setattr(body_host, "host_kernel_type", Mock(return_value=Mock(return_value=Mock())))
    port = SimpleNamespace(read_state=lambda: SimpleNamespace(joint_positions=(0.,) * 29))
    sock = None
    if prebound:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        sock.setblocking(False)
    try:
        kwargs = {"sock": sock} if prebound else {}
        host = body_host.BodyHost(port, bind="127.0.0.1:0", **kwargs)
        if prebound:
            assert host.sock is sock
        else:
            sock = host.sock
        assert not host.sock.getblocking()
        assert_reserved(f"127.0.0.1:{host.sock.getsockname()[1]}")
    finally:
        if sock is not None:
            sock.close()
