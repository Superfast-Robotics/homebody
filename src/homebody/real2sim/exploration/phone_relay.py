"""Relay one phone's SRT UDP session from USB or Wi-Fi to the workstation Ethernet link.

Forwards packets only; runs on the Jetson's Python 3.8.
"""
import argparse
import selectors
import signal
import socket
import time


def relay(listen, destination, idle_seconds=15.0):
    client = None
    last_client = 0.0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as inbound, \
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as outbound, \
            selectors.DefaultSelector() as selector:
        inbound.bind(listen)
        outbound.connect(destination)
        selector.register(inbound, selectors.EVENT_READ)
        selector.register(outbound, selectors.EVENT_READ)
        print("phone SRT relay listening on {}:{} -> {}:{}".format(*listen, *destination),
              flush=True)
        while True:
            for event, _ in selector.select(timeout=1):
                sock = event.fileobj
                try:
                    data, address = sock.recvfrom(65535)
                except ConnectionRefusedError:
                    continue  # the workstation receiver may not have started yet
                now = time.monotonic()
                if sock is inbound:
                    busy = client is not None and now - last_client < idle_seconds
                    if busy and address != client:
                        continue  # one active sender at a time
                    if address != client:
                        client = address
                        print("phone peer {}:{}".format(*client), flush=True)
                    last_client = now
                    try:
                        outbound.send(data)
                    except ConnectionRefusedError:
                        pass
                elif client is not None and now - last_client < idle_seconds:
                    inbound.sendto(data, client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", required=True, help="Jetson phone-facing IPv4 address")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--destination", required=True, help="workstation IPv4 address")
    parser.add_argument("--destination-port", type=int, default=9000)
    args = parser.parse_args()
    for address in (args.listen, args.destination):
        try:
            socket.inet_pton(socket.AF_INET, address)
        except OSError:
            parser.error("listen and destination must be IPv4 addresses")
    if not all(1 <= port <= 65535 for port in (args.port, args.destination_port)):
        parser.error("ports must be within 1..65535")

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        relay((args.listen, args.port), (args.destination, args.destination_port))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
