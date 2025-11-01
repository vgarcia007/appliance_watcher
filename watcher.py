#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-device appliance cycle monitor (Tasmota HTTP / Shelly RPC)
with per-device ntfy notifications + JSON status HTTP endpoint.

- Polls:
    * Tasmota: GET <base>/cm?cmnd=Status%2010 -> ENERGY.Power
    * Shelly Gen3 RPC: GET <base>/rpc/<method>?<params> -> apower (default)
- FSM per device with thresholds / timers
- ntfy notifications on FINISHED
- Thread-safe status registry exposed at /status (optional)
- Structured, thread-safe logging

Deps: requests
"""

import argparse
import csv
import datetime as dt
import json
import os
import signal
import sys
import threading
import time
from typing import Optional, Dict, Any, List, Tuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse as urlparse

import requests


# -------------------- utils / time --------------------

def now_mono() -> float:
    """Monotonic seconds for timers."""
    return time.monotonic()

def now_iso() -> str:
    """Wall-clock timestamp for logs/status."""
    return dt.datetime.now().isoformat(timespec="seconds")


# -------------------- thread-safe print --------------------

_print_lock = threading.Lock()

def tprint(msg: str, err: bool = False):
    with _print_lock:
        stream = sys.stderr if err else sys.stdout
        stream.write(msg + "\n")
        stream.flush()


# -------------------- payload parsing --------------------

def parse_power_tasmota(payload: dict) -> Optional[float]:
    """Extract ENERGY.Power (Watts) from Tasmota 'Status 10' or SENSOR payloads."""
    if isinstance(payload.get("ENERGY"), dict) and "Power" in payload["ENERGY"]:
        try:
            return float(payload["ENERGY"]["Power"])
        except Exception:
            return None
    status = payload.get("StatusSNS")
    if isinstance(status, dict):
        energy = status.get("ENERGY")
        if isinstance(energy, dict) and "Power" in energy:
            try:
                return float(energy["Power"])
            except Exception:
                return None
    return None

def parse_power_shelly_rpc(payload: dict, field: str = "apower") -> Optional[float]:
    """Extract active power from Shelly RPC response, default field 'apower'."""
    val = payload.get(field)
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


# -------------------- config models --------------------

class DeviceConfig:
    """Configuration per device."""

    def __init__(self, d: Dict[str, Any]):
        # Identity
        self.name: str = d["name"]

        # Source selection
        self.source: str = d.get("source", "tasmota").lower()  # "tasmota" | "shelly_rpc"
        self.base_url: str = d.get("base_url", d.get("tasmota_url", "")).rstrip("/")
        if not self.base_url:
            raise ValueError(f"{self.name}: base_url/tasmota_url required")

        # Optional device auth
        self.device_user: Optional[str] = d.get("device_user")
        self.device_pass: Optional[str] = d.get("device_pass")

        # Shelly RPC specifics
        self.shelly_rpc_method: str = d.get("shelly_rpc_method", "PM1.GetStatus")
        self.shelly_rpc_params: Dict[str, Any] = d.get("shelly_rpc_params", {"id": 0})
        self.shelly_power_field: str = d.get("shelly_power_field", "apower")

        # Polling
        self.poll_interval: float = float(d.get("poll_interval", 3.0))

        # Thresholds & timers
        self.start_threshold_w: float = float(d.get("start_threshold_w", 10.0))
        self.idle_threshold_w: float = float(d.get("idle_threshold_w", 3.0))
        self.start_confirm_seconds: int = int(d.get("start_confirm_seconds", 12))
        self.quiet_confirm_seconds: int = int(d.get("quiet_confirm_seconds", 60))
        self.min_run_seconds: int = int(d.get("min_run_seconds", 180))
        self.blip_tolerance_seconds: int = int(d.get("blip_tolerance_seconds", 6))
        self.cooldown_seconds: int = int(d.get("cooldown_seconds", 150))

        # ntfy (use env as fallback)
        self.ntfy_url: Optional[str] = d.get("ntfy_url")
        self.ntfy_topic: Optional[str] = d.get("ntfy_topic")
        self.ntfy_user: Optional[str] = d.get("ntfy_user") or os.getenv("NTFY_USER")
        self.ntfy_pass: Optional[str] = d.get("ntfy_pass") or os.getenv("NTFY_PASS")
        self.ntfy_title: str = d.get("ntfy_title", self.name)
        self.ntfy_icon: Optional[str] = d.get("ntfy_icon")
        self.ntfy_priority: int = int(d.get("ntfy_priority", 5))

        # CSV optional
        self.csv_log_path: Optional[str] = d.get("csv_log_path")
        self.log_quiet_progress: bool = bool(d.get("log_quiet_progress", False))


class GlobalConfig:
    def __init__(self, args):
        self.devices_file = args.devices_file
        self.status_http: Optional[Tuple[str, int]] = None
        if args.status_http:
            host, port = args.status_http.split(":")
            self.status_http = (host, int(port))
        self.heartbeat_seconds: Optional[int] = args.heartbeat_seconds
        self.shutdown = False


# -------------------- status registry + HTTP --------------------

class StatusRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}

    def update(self, key: str, **fields):  # was: name: str
        with self._lock:
            cur = self._data.get(key, {})
            cur.update(fields)
            self._data[key] = cur

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data, default=str))



_status_registry = StatusRegistry()


class StatusHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse.urlparse(self.path)
        if parsed.path == "/status":
            self._send_json({
                "ts": now_iso(),
                "devices": _status_registry.snapshot()
            })
        else:
            self._send_json({"error": "not found"}, code=404)

    def log_message(self, fmt, *args):
        # Silence default HTTP server logging to keep stdout clean
        pass


def start_status_http_server(bind: Tuple[str, int]):
    host, port = bind
    httpd = ThreadingHTTPServer((host, port), StatusHandler)
    tprint(f"[boot] status http server on http://{host}:{port}/status")
    httpd.serve_forever()


# -------------------- notifications --------------------

def send_ntfy(dev: DeviceConfig, text: str):
    """Send message to ntfy topic with optional Basic-Auth."""
    if not dev.ntfy_url or not dev.ntfy_topic:
        tprint(f"[{dev.name}] [ntfy] missing url/topic, skipping")
        return
    url = f"{dev.ntfy_url.rstrip('/')}/{dev.ntfy_topic}"
    headers = {
        "Title": dev.ntfy_title,
        "Priority": str(dev.ntfy_priority),
    }
    if dev.ntfy_icon:
        headers["Icon"] = dev.ntfy_icon

    auth = (dev.ntfy_user, dev.ntfy_pass) if (dev.ntfy_user and dev.ntfy_pass) else None
    try:
        r = requests.post(url, headers=headers, data=text.encode("utf-8"), auth=auth, timeout=10)
        if r.status_code >= 300:
            tprint(f"[{dev.name}] [ntfy] HTTP {r.status_code}: {r.text}", err=True)
    except Exception as e:
        tprint(f"[{dev.name}] [ntfy] error: {e}", err=True)


# -------------------- finite state machine --------------------

class WasherState:
    """Per-device FSM with status updates."""

    def __init__(self, conf: DeviceConfig):
        self.c = conf
        self.state = "IDLE"
        self.last_above_start: Optional[float] = None
        self.last_below_idle: Optional[float] = None
        self.run_started_at: Optional[float] = None
        self.cooldown_until: float = 0.0
        self.last_power_w: float = 0.0
        self.last_state_change_iso: str = now_iso()
        self._lock = threading.Lock()
        self.cycle_data: List[Tuple[float, float]] = []  # (mono_time, power)
        self._update_registry()

    # --- helpers ---

    def _log(self, msg: str):
        tprint(f"[{self.c.name}] {msg}")

    def _update_registry(self):
        cooldown_left = max(0, int(self.cooldown_until - now_mono())) if self.cooldown_until else 0
        _status_registry.update(
            self.c.name,
            name=self.c.name,
            source=self.c.source,
            base_url=self.c.base_url,
            state=self.state,
            last_power_w=round(self.last_power_w, 1),
            last_state_change=self.last_state_change_iso,
            run_started_at=self._iso_or_none(self.run_started_at),
            cooldown_remaining_s=cooldown_left,
            thresholds=dict(
                start_threshold_w=self.c.start_threshold_w,
                idle_threshold_w=self.c.idle_threshold_w,
                start_confirm_s=self.c.start_confirm_seconds,
                quiet_confirm_s=self.c.quiet_confirm_seconds,
                min_run_s=self.c.min_run_seconds,
                blip_tol_s=self.c.blip_tolerance_seconds,
            ),
            updated_at=now_iso(),
        )

    @staticmethod
    def _iso_or_none(tmono: Optional[float]) -> Optional[str]:
        # Not mapping monotonic to wall clock; we just return ISO when set, else None.
        return now_iso() if tmono is not None else None

    def _set_state(self, new_state: str):
        if new_state != self.state:
            self.state = new_state
            self.last_state_change_iso = now_iso()
            self._log(f"state -> {new_state}")
            self._update_registry()

    # --- transitions ---

    def _enter_running(self):
        self.run_started_at = now_mono()
        self.last_below_idle = None
        self.cycle_data = []
        self._set_state("RUNNING")

    def _enter_quiet_timer(self):
        self.last_below_idle = now_mono()
        self._set_state("QUIET_TIMER")

    def _finish_and_notify(self):
        dur_s = int((now_mono() - self.run_started_at)) if self.run_started_at else 0
        mins, secs = divmod(dur_s, 60)
        msg = f"{self.c.name} ist fertig. Laufzeit: {mins}m {secs}s."
        self._set_state("FINISHED")
        self._log("sending ntfy")
        send_ntfy(self.c, msg)
        self._log_complete_cycle(dur_s)
        # enter cooldown
        self.cooldown_until = now_mono() + self.c.cooldown_seconds
        self.last_above_start = None
        self.last_below_idle = None
        self.run_started_at = None
        self._set_state("COOLDOWN")

    def _log_complete_cycle(self, duration_s: int):
        if not self.cycle_data:
            return
        
        log_dir = "/app/logs"
        finished_time = dt.datetime.now()
        filename = f"{self.c.name.replace(' ', '_')}_{finished_time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        filepath = os.path.join(log_dir, filename)

        try:
            cycle_entry = {
                "device": self.c.name,
                "started_at": self._iso_or_none(self.run_started_at),
                "finished_at": finished_time.isoformat(timespec="seconds"),
                "duration_s": duration_s,
                "data": [{"time_offset_s": round(t - self.run_started_at, 1), "power_w": round(p, 1)} for t, p in self.cycle_data]
            }
            os.makedirs(log_dir, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(cycle_entry, f, ensure_ascii=False, indent=2)
            self._log(f"cycle data saved to {filepath}")
        except Exception as e:
            self._log(f"cycle log error: {e}")

    # --- public ---

    def update_power(self, power_w: float):
        with self._lock:
            self.last_power_w = power_w
            # reflect power in registry
            self._update_registry()

            t = now_mono()

            if self.state == "COOLDOWN":
                if t >= self.cooldown_until and power_w <= self.c.idle_threshold_w:
                    self._set_state("IDLE")
                return

            if self.state == "IDLE":
                if power_w >= self.c.start_threshold_w:
                    if self.last_above_start is None:
                        self.last_above_start = t
                    if t - self.last_above_start >= self.c.start_confirm_seconds:
                        self._enter_running()
                else:
                    self.last_above_start = None
                return

            if self.state == "RUNNING":
                self.cycle_data.append((t, power_w))
                if power_w <= self.c.idle_threshold_w:
                    if self.last_below_idle is None:
                        self.last_below_idle = t
                    if t - self.last_below_idle >= self.c.blip_tolerance_seconds:
                        self._enter_quiet_timer()
                else:
                    self.last_below_idle = None
                return

            if self.state == "QUIET_TIMER":
                if power_w <= self.c.idle_threshold_w:
                    elapsed_quiet = t - (self.last_below_idle or t)
                    if self.c.log_quiet_progress:
                        self._log(f"quiet={int(elapsed_quiet)}s/{self.c.quiet_confirm_seconds}s (power={power_w:.1f} W)")
                    if elapsed_quiet >= self.c.quiet_confirm_seconds:
                        if self.run_started_at and (t - self.run_started_at) >= self.c.min_run_seconds:
                            self._finish_and_notify()
                        else:
                            self._log("finish rejected (short run); -> IDLE")
                            self.last_above_start = None
                            self.last_below_idle = None
                            self.run_started_at = None
                            self._set_state("IDLE")
                else:
                    if power_w >= self.c.start_threshold_w:
                        self._log("quiet aborted (power up)")
                        self.last_below_idle = None
                        self._set_state("RUNNING")
                return


# -------------------- polling threads --------------------

def maybe_append_csv(dev: DeviceConfig, power: float):
    if not dev.csv_log_path:
        return
    try:
        os.makedirs(os.path.dirname(dev.csv_log_path), exist_ok=True)
        with open(dev.csv_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter=";").writerow([now_iso(), f"{power:.1f}"])
    except Exception as e:
        tprint(f"[{dev.name}] csv log error: {e}", err=True)

def poll_device_loop(dev: DeviceConfig, gcfg: GlobalConfig, heartbeat_seconds: Optional[int]):
    fsm = WasherState(dev)
    session = requests.Session()
    auth = (dev.device_user, dev.device_pass) if (dev.device_user and dev.device_pass) else None
    timeout = 5

    if dev.source == "tasmota":
        endpoint = f"{dev.base_url}/cm"
        params = {"cmnd": "Status 10"}

        def fetch_power() -> Optional[float]:
            r = session.get(endpoint, params=params, auth=auth, timeout=timeout)
            return parse_power_tasmota(r.json())

    elif dev.source == "shelly_rpc":
        endpoint = f"{dev.base_url}/rpc/{dev.shelly_rpc_method}"
        params = {**(dev.shelly_rpc_params or {})}

        def fetch_power() -> Optional[float]:
            r = session.get(endpoint, params=params, auth=auth, timeout=timeout)
            return parse_power_shelly_rpc(r.json(), dev.shelly_power_field)

    else:
        tprint(f"[{dev.name}] unknown source '{dev.source}'", err=True)
        return

    last_heartbeat = now_mono()

    while not gcfg.shutdown:
        try:
            pwr = fetch_power()
            if pwr is not None:
                tprint(f"[{dev.name}] POWER={pwr:.1f} W")
                maybe_append_csv(dev, pwr)
                fsm.update_power(pwr)
            else:
                tprint(f"[{dev.name}] WARN: couldn't parse power", err=True)
        except Exception as e:
            tprint(f"[{dev.name}] poll error: {e}", err=True)

        # optional heartbeat line with current state
        if heartbeat_seconds:
            if (now_mono() - last_heartbeat) >= heartbeat_seconds:
                snap = _status_registry.snapshot().get(dev.name, {})
                tprint(f"[dev.name] HEARTBEAT state={snap.get('state')} power={snap.get('last_power_w')}W")
                last_heartbeat = now_mono()

        slept = 0.0
        step = 0.2
        while slept < dev.poll_interval and not gcfg.shutdown:
            time.sleep(step)
            slept += step


# -------------------- boot --------------------

def load_devices(path: str) -> List[DeviceConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("devices.json must be a list of objects")
    return [DeviceConfig(d) for d in raw]

def build_argparser():
    ap = argparse.ArgumentParser(description="Multi-device cycle detector (Tasmota/Shelly RPC) + ntfy + HTTP status")
    ap.add_argument("--devices-file", default=os.getenv("DEVICES_FILE", "/app/devices.json"),
                    help="Path to devices JSON file")
    ap.add_argument("--status-http", default=os.getenv("STATUS_HTTP", ""),  # e.g. "0.0.0.0:8080"
                    help="Bind host:port for JSON status server (empty to disable)")
    ap.add_argument("--heartbeat-seconds", type=int, default=int(os.getenv("HEARTBEAT_SECONDS", "0")),
                    help="Log heartbeat line every N seconds (0=off)")
    return ap

def main():
    args = build_argparser().parse_args()
    gcfg = GlobalConfig(args)

    try:
        devices = load_devices(gcfg.devices_file)
    except Exception as e:
        tprint(f"Failed to load devices: {e}", err=True)
        sys.exit(2)

    if not devices:
        tprint("No devices defined", err=True)
        sys.exit(2)

    # Start HTTP status server if requested
    if gcfg.status_http:
        th = threading.Thread(target=start_status_http_server, args=(gcfg.status_http,), daemon=True)
        th.start()

    # Start device threads
    threads = []
    for d in devices:
        t = threading.Thread(target=poll_device_loop, args=(d, gcfg, gcfg.heartbeat_seconds), daemon=True)
        t.start()
        threads.append(t)
        tprint(f"[boot] started device thread: {d.name} ({d.source})")

    # Graceful shutdown
    def _sig_handler(signum, _):
        tprint(f"[boot] signal {signum} -> shutdown")
        gcfg.shutdown = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _sig_handler)

    try:
        while not gcfg.shutdown:
            time.sleep(0.5)
    except KeyboardInterrupt:
        gcfg.shutdown = True

    tprint("[boot] waiting for threads...")
    for t in threads:
        t.join(timeout=2)
    tprint("[boot] bye.")


if __name__ == "__main__":
    main()
