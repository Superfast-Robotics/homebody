# iPhone camera and robot sensor setup

Reference path: **iPhone → USB network → G1 Jetson RTSP relay → Ethernet → workstation**, RTSP over TCP, MediaMTX
v1.21.1, H.264 at 1280×720 and 30 fps. The phone gives appearance and semantics, never obstacle clearance.

## Hardware and network

- Use the supported AMO locomotion assets and an iPhone with a rear ultra-wide camera (reference: iPhone 15 Plus),
  landscape on the upper chest in the [printable mount](../hardware/g1_iphone_mount.stl) (not a measured extrinsic).
- Keep the workstation's Internet for the model provider and give its robot link an unused `192.168.123.x` address
  with no default gateway; `--iface` (`iface` in `[robot]`) chooses the interface, not a replacement robot subnet.
- In `[robot]`, `ssh_target` is the robot SSH account and address and `ssh_password` defaults to the factory `123`;
  `HOMEBODY_G1_SSH_PASSWORD` overrides it for a changed password.
- Record from the actual network (USB DHCP leases differ per device): `JETSON_USB_IP` (Jetson on the phone's USB
  network, where the phone publishes), `PHONE_USB_IP` (phone source there; relay publication allowlist),
  `JETSON_LAN_IP` (Jetson on robot Ethernet, where the workstation reads), `WORKSTATION_LAN_IP` (reader allowlist).
- On the Jetson install `usbmuxd`, `libimobiledevice-utils` and `usbutils`, connect a USB-C to USB-C data cable to a
  USB host port, unlock the phone, enable **Personal Hotspot**, reconnect, accept **Trust This Computer**
  ([Apple's trust dialog and reset procedure](https://support.apple.com/en-us/109054)) and inspect on the Jetson:

```bash
lsusb -d 05ac:
idevicepair pair
ip -br link
ip -br -4 address
```

- Linux needs the `ipheth` driver: if `modinfo ipheth` fails and no phone interface appears, build it for that exact
  Jetson kernel and JetPack ([NVIDIA's kernel customization guide](https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/SD/Kernel/KernelCustomization.html))
  rather than copying another machine's module. Use DHCP on the phone interface, **never the default route** and
  without its automatic DNS in NetworkManager; verify its lease and peer (commonly the USB gateway).

## Provision the RTSP relay once

Jetson administration only, starting no locomotion: in a copy of [`hardware/g1/phone/`](../hardware/g1/phone/) (with
`curl` and a CA bundle) install the pinned [MediaMTX v1.21.1 release](https://github.com/bluenviron/mediamtx/releases/tag/v1.21.1) (`aarch64` uses `linux_arm64`):

```bash
mkdir -p download
curl --fail --location --output download/mediamtx.tar.gz \
  https://github.com/bluenviron/mediamtx/releases/download/v1.21.1/mediamtx_v1.21.1_linux_arm64.tar.gz
printf '%s  %s\n' 6a3aa635fb60ea9b8d566ec306f0a42ff1b6b52a3942bc2baffbe55880d4c3dd download/mediamtx.tar.gz | sha256sum --check -
tar -xzf download/mediamtx.tar.gz -C download mediamtx LICENSE
sudo install -d -m 755 /opt/homebody-phone /etc/homebody-phone
sudo install -m 755 download/mediamtx /opt/homebody-phone/mediamtx-v1.21.1
sudo install -m 644 download/LICENSE /opt/homebody-phone/LICENSE
cp mediamtx.yml phone.local.yml
# Edit phone.local.yml: the first 127.0.0.1 becomes PHONE_USB_IP, the second WORKSTATION_LAN_IP; leave no others.
# Check the configuration without opening a listener, then install and start the service.
/opt/homebody-phone/mediamtx-v1.21.1 --validate-conf=phone.local.yml
sudo install -m 644 phone.local.yml /etc/homebody-phone/mediamtx.yml
sudo install -m 644 homebody-phone-rtsp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now homebody-phone-rtsp.service
systemctl status homebody-phone-rtsp.service --no-pager
```

It allows one publisher at path `iphone`, RTSP over TCP on 8554 and no other listener, management API or recording
([IP permissions](https://mediamtx.org/docs/references/configuration-file)): a trusted local network configuration,
not an Internet-facing authenticated streaming service. Open TCP 8554 to the phone and workstation and let the
launcher's SSH connection restart the service (it runs as a temporary unprivileged user). With `--iphone-relay-service`
setup starts/reuses it and each managed launch restarts it (the phone app must reconnect); it outlives exploration.

## Verify video before starting the robot workflow

Publish with [Larix Broadcaster for iOS](https://softvelum.com/larix/ios/) or another app: rear **0.5× ultra-wide**,
landscape, H.264, **1280×720 at 30 fps**, HDR off, no overlay, RTSP over TCP to `rtsp://JETSON_USB_IP:8554/iphone`,
no session or license limit; check view, level and clearance, and record crop/stabilization. On the workstation:

```bash
export PHONE_STREAM_URL="rtsp://JETSON_LAN_IP:8554/iphone"
export PHONE_RELAY_SERVICE=homebody-phone-rtsp.service
ffprobe -v error -rtsp_transport tcp -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 "$PHONE_STREAM_URL"
ffplay -rtsp_transport tcp "$PHONE_STREAM_URL"
```

The picture must change with the scene. `404 Not Found` on `/iphone` means no publisher (check the app and
`journalctl -u homebody-phone-rtsp.service -n 40 --no-pager`); refusal is a listener/address/firewall problem, a
permission error the allowlists. Then run the [exploration quick run](exploration.md#quick-run-after-installation)
with `--iphone-input "$PHONE_STREAM_URL"` and `--iphone-relay-service "$PHONE_RELAY_SERVICE"`: its receiver needs
fresh RGB (receipt time catches a stalled feed, not exposure delay or a frozen retransmitted picture), gives the model
5 fps at 960 pixels wide and, with recording, keeps native video. Calibration, clock alignment and the roughly three
second delay are covered in [Real-to-sim](real2sim.md#audit-a-closed-run-first).

**Alternatives.** Any FFmpeg-readable source can replace the relay; omit `--iphone-relay-service` so externally
managed video is left alone. With [Moblin](https://github.com/eerimoq/moblin), SRT can reach a workstation listener,
`--iphone-input 'srt://WORKSTATION_LAN_IP:9000?mode=listener&latency=200000'` (FFmpeg with SRT, UDP connectivity), or
via [`phone_relay.py`](../src/homebody/real2sim/exploration/phone_relay.py), which forwards one SRT session from the
Jetson USB link. Never mix the RTSP publication URL with an SRT receiver; check quality, latency and native recording.
