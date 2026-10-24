"""The dashboard server over a real socket: its page, its WebSocket protocol and what it refuses."""
import asyncio
import json
import struct
import urllib.error
import urllib.request

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from homebody.ui.dashboard.server import Hub, frame_message, static_files
from homebody.ui.dashboard.views import History


@pytest.fixture
def hub():
    commands = []
    history = History()
    history.add(5., b"\xff\xd8stored")

    def on_command(request):
        commands.append(request)
        return {"ok": request.get("type") == "submit", "message": "noted"}
    server = Hub("127.0.0.1", 0, snapshot=lambda: {"status": "Ready"}, on_command=on_command,
                 history=history).start()
    server.commands = commands
    yield server
    server.close()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def test_the_page_its_files_and_stored_frames_are_served_and_nothing_else(hub):
    status, headers, body = get(hub.url + "/")
    assert status == 200 and b"Tell the agent what to do..." in body
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert get(hub.url + "/js/app.js")[0] == 200
    assert get(hub.url + "/history/1.jpg")[2] == b"\xff\xd8stored"
    assert get(hub.url + "/history/9.jpg")[0] == 404
    for path in ("/../server.py", "/%2e%2e/server.py", "/server.py", "/js/../../store.py"):
        assert get(hub.url + path)[0] == 404


def test_every_page_file_is_listed_once():
    files = static_files()
    assert {"/", "/index.html", "/tokens.css", "/dashboard.css", "/js/app.js"} <= set(files)
    assert files["/js/app.js"][0].startswith("text/javascript")


async def session(hub, *messages, origin=None):
    headers = {"Origin": origin} if origin else {}
    async with connect(hub.url.replace("http", "ws") + "/ws", additional_headers=headers) as ws:
        received = [json.loads(await ws.recv())]
        for message in messages:
            await ws.send(message)
            reply = {}
            while reply.get("type") != "reply":  # state broadcasts may arrive first
                reply = json.loads(await asyncio.wait_for(ws.recv(), 5))
            received.append(reply)
        return received


def test_state_on_connect_and_replies_to_commands(hub):
    state, reply, malformed = asyncio.run(session(
        hub, json.dumps({"id": 4, "type": "submit", "text": "Pick"}), "not json"))
    assert state == {"type": "state", "state": {"status": "Ready"}}
    assert reply == {"type": "reply", "id": 4, "ok": True, "message": "noted"}
    assert malformed["ok"] is False and hub.commands == [{"id": 4, "type": "submit", "text": "Pick"}]


def test_a_page_from_another_origin_cannot_open_the_socket(hub):
    with pytest.raises(InvalidStatus):
        asyncio.run(session(hub, origin="http://evil.example"))
    asyncio.run(session(hub, origin=hub.url))


def test_frames_reach_only_the_clients_watching_that_view(hub):
    async def watch():
        async with connect(hub.url.replace("http", "ws") + "/ws") as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "view", "view": "depth"}))
            await asyncio.sleep(.1)
            assert hub.wanted() == {"depth"}
            hub.frame("robot", {"t": 1.}, b"robot")
            hub.frame("depth", {"t": 2.}, b"depth")
            return await asyncio.wait_for(ws.recv(), 5)
    message = asyncio.run(watch())
    length, = struct.unpack(">I", message[:4])
    assert json.loads(message[4:4 + length]) == {"t": 2., "view": "depth"} and message[4 + length:] == b"depth"
    assert frame_message({"view": "x"}, b"jpg").endswith(b"jpg")


def test_a_taken_port_is_reported():
    first = Hub("127.0.0.1", 0, snapshot=dict, on_command=dict).start()
    try:
        with pytest.raises(OSError):
            Hub("127.0.0.1", first.port, snapshot=dict, on_command=dict).start()
    finally:
        first.close()
