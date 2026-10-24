// The page's one WebSocket: state snapshots, camera frames and telemetry in; commands out.
const RETRY = [500, 1000, 2000, 5000];
const REPLY_TIMEOUT = 8000;

export class Link {
  /**
   * @param {{onState: Function, onFrame: Function, onTelemetry: Function, onConnection: Function}} handlers
   */
  constructor(handlers) {
    this.handlers = handlers;
    this.socket = null;
    this.connected = false;
    this.attempt = 0;
    this.nextId = 1;
    this.waiting = new Map();
    this.view = "robot";
    this.offset = 0; // server clock minus this browser's, in seconds
  }

  /** Server time now, in seconds: the one clock every panel uses. */
  now() {
    return Date.now() / 1000 + this.offset;
  }

  connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/ws`);
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    socket.addEventListener("open", () => {
      this.connected = true;
      this.attempt = 0;
      socket.send(JSON.stringify({ type: "view", view: this.view }));
      this.handlers.onConnection(true);
    });
    socket.addEventListener("message", (event) => this.receive(event.data));
    socket.addEventListener("close", () => this.closed(socket));
    socket.addEventListener("error", () => socket.close());
  }

  closed(socket) {
    if (socket !== this.socket) return;
    this.connected = false;
    for (const [, waiter] of this.waiting) waiter({ ok: false, message: "Disconnected before the session replied." });
    this.waiting.clear();
    this.handlers.onConnection(false);
    const delay = RETRY[Math.min(this.attempt, RETRY.length - 1)];
    this.attempt += 1;
    setTimeout(() => this.connect(), delay);
  }

  receive(data) {
    if (data instanceof ArrayBuffer) {
      const length = new DataView(data).getUint32(0);
      const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 4, length)));
      this.handlers.onFrame(meta, new Blob([new Uint8Array(data, 4 + length)], { type: "image/jpeg" }));
      return;
    }
    let message;
    try {
      message = JSON.parse(data);
    } catch {
      return;
    }
    if (message.type === "state") {
      this.offset = message.state.now - Date.now() / 1000;
      this.handlers.onState(message.state);
    } else if (message.type === "telemetry") {
      this.handlers.onTelemetry(message);
    } else if (message.type === "reply" && this.waiting.has(message.id)) {
      this.waiting.get(message.id)(message);
      this.waiting.delete(message.id);
    }
  }

  /** Send a command; resolves with the session's reply {ok, message}, never rejects. */
  request(type, payload = {}) {
    if (!this.connected) return Promise.resolve({ ok: false, message: "Not connected to the session." });
    const id = this.nextId++;
    return new Promise((resolve) => {
      this.waiting.set(id, resolve);
      this.socket.send(JSON.stringify({ id, type, ...payload }));
      setTimeout(() => {
        if (this.waiting.delete(id)) resolve({ ok: false, message: "The session did not reply in time." });
      }, REPLY_TIMEOUT);
    });
  }

  setView(view) {
    this.view = view;
    if (this.connected) this.socket.send(JSON.stringify({ type: "view", view }));
  }
}
