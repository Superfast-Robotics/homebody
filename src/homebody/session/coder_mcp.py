"""The coding sub-agent's robot tools as a stdio MCP server: Codex starts it, and each tool call is
forwarded as one JSON line to the HomeBody session listening on SOCKET, which runs it on the live
robot and answers with one JSON line. Usage: python -m homebody.session.coder_mcp SOCKET
"""
import json
import socket
import sys

ARGUMENTS = {"type": "object", "properties": {"arguments": {
    "type": "object", "description": "The skill's arguments, as the agent would pass them."}},
    "required": ["arguments"]}
TOOLS = [
    {"name": "check", "inputSchema": ARGUMENTS, "description": (
        "Load skills/<target>.py (and its lib imports) and dry-run its prepare() on the robot's current "
        "observation without moving. Reports load errors, the refusal code prepare raised, or a summary "
        "of the plan. Required before execute, for the exact code execute will run.")},
    {"name": "execute", "inputSchema": ARGUMENTS, "description": (
        "Run the checked skills/<target>.py on the real robot with these arguments: the robot moves. "
        "Returns the skill's result, the hands' state and the base pose, and refreshes observation/ with "
        "the view afterwards.")},
]


def forward(path: str, tool: str, arguments: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(path)
        connection.sendall((json.dumps({"tool": tool, "arguments": arguments}) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = connection.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data or b'{"error": "the robot session closed the connection"}')


def reply(message_id, result=None, error=None):
    body = {"jsonrpc": "2.0", "id": message_id}
    body.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(body) + "\n")
    sys.stdout.flush()


def handle(path: str, message: dict) -> None:
    method, message_id = message.get("method"), message.get("id")
    if message_id is None:
        return  # a notification
    if method == "initialize":
        version = message.get("params", {}).get("protocolVersion", "2025-06-18")
        reply(message_id, {"protocolVersion": version, "capabilities": {"tools": {}},
                           "serverInfo": {"name": "robot", "version": "1"}})
    elif method == "tools/list":
        reply(message_id, {"tools": TOOLS})
    elif method == "tools/call":
        params = message.get("params", {})
        name, arguments = params.get("name"), (params.get("arguments") or {}).get("arguments", {})
        if name not in ("check", "execute"):
            reply(message_id, error={"code": -32602, "message": f"unknown tool {name!r}"})
            return
        try:
            answer = forward(path, name, arguments if isinstance(arguments, dict) else {})
        except OSError as error:
            answer = {"error": f"robot session unreachable: {error}"}
        reply(message_id, {"content": [{"type": "text", "text": json.dumps(answer, indent=1)}],
                           "isError": "error" in answer})
    elif method == "ping":
        reply(message_id, {})
    else:
        reply(message_id, error={"code": -32601, "message": f"method {method!r} not supported"})


def main() -> None:
    path = sys.argv[1]
    for line in sys.stdin:
        if line.strip():
            handle(path, json.loads(line))


if __name__ == "__main__":
    main()
