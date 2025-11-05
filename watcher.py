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

        # Cycle logging
        self.log_cycles: bool = bool(d.get("log_cycles", False))

        # Program detection rules (optional - fallback)
        self.program_rules: List[Dict[str, Any]] = d.get("program_rules", [])
        # Machine learning settings
        self.enable_auto_learning: bool = bool(d.get("enable_auto_learning", True))
        self.min_cycles_for_learning: int = int(d.get("min_cycles_for_learning", 3))


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
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html_content, code=200):
        raw = html_content.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse.urlparse(self.path)
        if parsed.path == "/status":
            self._send_json({
                "ts": now_iso(),
                "devices": _status_registry.snapshot()
            })
        elif parsed.path == "/" or parsed.path == "/index.html":
            try:
                # Try to read index.html from the same directory as this script
                script_dir = os.path.dirname(os.path.abspath(__file__))
                index_path = os.path.join(script_dir, "index.html")
                with open(index_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                self._send_html(html_content)
            except FileNotFoundError:
                self._send_json({"error": "index.html not found"}, code=404)
            except Exception as e:
                self._send_json({"error": f"Failed to load index.html: {str(e)}"}, code=500)
        else:
            self._send_json({"error": "not found"}, code=404)

    def do_OPTIONS(self):
        # Handle preflight CORS requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

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
        # Program prediction
        self.predicted_program: Optional[str] = None
        self.prediction_confidence: float = 0.0
        # Progress estimation
        self.estimated_total_duration_s: Optional[int] = None
        self.estimated_remaining_s: Optional[int] = None
        self.progress_percent: Optional[float] = None
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
            predicted_program=self.predicted_program,
            prediction_confidence=round(self.prediction_confidence, 2),
            estimated_total_duration_s=self.estimated_total_duration_s,
            estimated_remaining_s=self.estimated_remaining_s,
            progress_percent=round(self.progress_percent, 1) if self.progress_percent is not None else None,
            updated_at=now_iso(),
        )

    @staticmethod
    def _iso_or_none(tmono: Optional[float]) -> Optional[str]:
        if tmono is None:
            return None
        # Convert monotonic time to wall clock time
        # tmono is when the event happened, now_mono() is current monotonic time
        # Subtract the difference from current wall clock time
        time_ago_seconds = now_mono() - tmono
        actual_time = dt.datetime.now() - dt.timedelta(seconds=time_ago_seconds)
        return actual_time.isoformat(timespec="seconds")

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
        # Reset program prediction and progress
        self.predicted_program = None
        self.prediction_confidence = 0.0
        self.estimated_total_duration_s = None
        self.estimated_remaining_s = None
        self.progress_percent = None
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
        self._log(f"_log_complete_cycle called: log_cycles={self.c.log_cycles}, cycle_data_length={len(self.cycle_data)}")
        if not self.c.log_cycles:
            self._log("cycle logging disabled for this device")
            return
        
        log_dir = "/app/logs"
        finished_time = dt.datetime.now()
        filename = f"{self.c.name.replace(' ', '_')}_{finished_time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        filepath = os.path.join(log_dir, filename)
        self._log(f"attempting to write cycle log to: {filepath}")

        try:
            # Calculate final statistics
            powers = [p for _, p in self.cycle_data] if self.cycle_data else []
            avg_power = sum(powers) / len(powers) if powers else 0
            max_power = max(powers) if powers else 0
            
            cycle_entry = {
                "device": self.c.name,
                "started_at": self._iso_or_none(self.run_started_at),
                "finished_at": finished_time.isoformat(timespec="seconds"),
                "duration_s": duration_s,
                "predicted_program": self.predicted_program,
                "prediction_confidence": round(self.prediction_confidence, 2),
                "statistics": {
                    "avg_power_w": round(avg_power, 1),
                    "max_power_w": round(max_power, 1),
                    "data_points": len(self.cycle_data)
                },
                "data": [{"time_offset_s": round(t - self.run_started_at, 1), "power_w": round(p, 1)} for t, p in self.cycle_data] if self.cycle_data else [],
                "note": "Incomplete cycle data - container may have been restarted during cycle" if not self.cycle_data else None
            }
            os.makedirs(log_dir, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(cycle_entry, f, ensure_ascii=False, indent=2)
            self._log(f"SUCCESS: cycle data saved to {filepath}")
        except Exception as e:
            self._log(f"ERROR: cycle log failed: {e}")
            import traceback
            self._log(f"ERROR: traceback: {traceback.format_exc()}")

    def _predict_program(self):
        """Predict program using ML clustering or fallback to rules."""
        try:
            if not self.cycle_data or len(self.cycle_data) < 10:
                return
                
            # Try ML-based prediction first
            if self.c.enable_auto_learning:
                self._predict_program_ml()
            
            # Fallback to rule-based prediction
            elif self.c.program_rules:
                self._predict_program_rules()
                
        except Exception as e:
            self._log(f"Program prediction disabled due to error: {e}")
            return
        
        # Extract recent power values (last 5 minutes or 100 data points)
        recent_data = self.cycle_data[-100:]
        powers = [p for _, p in recent_data]
        
        if not powers:
            return
        
        # Calculate basic statistics
        avg_power = sum(powers) / len(powers)
        max_power = max(powers)
        min_power = min(powers)
        
        # Check each rule
        best_match = None
        best_confidence = 0.0
        
        for rule in self.c.program_rules:
            try:
                confidence = self._evaluate_rule(rule, avg_power, max_power, min_power, powers)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = rule.get("name", "Unknown")
            except Exception as e:
                self._log(f"Error evaluating rule {rule.get('name', 'Unknown')}: {e}")
                continue
        
        # Update prediction if better than current
        if best_confidence > self.prediction_confidence:
            self.predicted_program = best_match
            self.prediction_confidence = best_confidence
            self._log(f"Program prediction: {best_match} (confidence: {best_confidence:.0%})")
            
            # Update progress estimation
            self._update_progress_estimation()
            self._update_registry()

    def _predict_program_ml(self):
        """ML-based program prediction using historical data clustering."""
        # Extract features from current cycle
        current_features = self._extract_cycle_features()
        if not current_features:
            return
            
        # Load and analyze historical cycles
        historical_data = self._load_historical_cycles()
        if len(historical_data) < self.c.min_cycles_for_learning:
            self._log(f"Not enough historical data for ML ({len(historical_data)}/{self.c.min_cycles_for_learning})")
            return
            
        # Perform clustering and prediction
        predicted_profile = self._cluster_and_predict(current_features, historical_data)
        if predicted_profile:
            confidence = 0.8  # ML-based predictions get high confidence
            if confidence > self.prediction_confidence:
                self.predicted_program = predicted_profile
                self.prediction_confidence = confidence
                self._log(f"ML prediction: {predicted_profile} (confidence: {confidence:.0%})")

    def _extract_cycle_features(self) -> Optional[Dict[str, float]]:
        """Extract statistical features from current cycle for ML."""
        if not self.cycle_data:
            return None
            
        powers = [p for _, p in self.cycle_data[-100:]]  # Last 100 points
        if len(powers) < 10:
            return None
            
        # Calculate comprehensive features
        avg_power = sum(powers) / len(powers)
        max_power = max(powers)
        min_power = min(powers)
        
        # Statistical features
        import statistics
        power_std = statistics.stdev(powers) if len(powers) > 1 else 0
        power_median = statistics.median(powers)
        
        # Pattern features
        high_power_100 = sum(1 for p in powers if p >= 100) / len(powers)
        high_power_200 = sum(1 for p in powers if p >= 200) / len(powers)
        high_power_400 = sum(1 for p in powers if p >= 400) / len(powers)
        
        # Stability features
        variation_coeff = power_std / avg_power if avg_power > 0 else 0
        
        return {
            'avg_power': avg_power,
            'max_power': max_power,
            'min_power': min_power,
            'std_power': power_std,
            'median_power': power_median,
            'high_power_100_ratio': high_power_100,
            'high_power_200_ratio': high_power_200,
            'high_power_400_ratio': high_power_400,
            'variation_coefficient': variation_coeff,
        }

    def _load_historical_cycles(self) -> List[Dict[str, Any]]:
        """Load all historical cycle data for ML analysis."""
        import glob
        import os
        
        cycles = []
        log_pattern = f"/app/logs/{self.c.name.replace(' ', '_')}_*.json"
        
        try:
            for log_file in glob.glob(log_pattern):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Extract features from historical cycle
                    if 'data' in data and data['data']:
                        powers = [entry['power_w'] for entry in data['data']]
                        features = self._calculate_features_from_powers(powers)
                        
                        if features:
                            cycle_info = {
                                'features': features,
                                'duration': data.get('duration_s', 0),
                                'file': log_file,
                                'predicted_program': data.get('predicted_program'),
                                'timestamp': data.get('finished_at')
                            }
                            cycles.append(cycle_info)
                            
                except Exception:
                    continue
        except Exception:
            pass
            
        return cycles

    def _calculate_features_from_powers(self, powers: List[float]) -> Optional[Dict[str, float]]:
        """Calculate features from a list of power values."""
        if len(powers) < 10:
            return None
            
        avg_power = sum(powers) / len(powers)
        max_power = max(powers)
        min_power = min(powers)
        
        import statistics
        power_std = statistics.stdev(powers) if len(powers) > 1 else 0
        power_median = statistics.median(powers)
        
        high_power_100 = sum(1 for p in powers if p >= 100) / len(powers)
        high_power_200 = sum(1 for p in powers if p >= 200) / len(powers)
        high_power_400 = sum(1 for p in powers if p >= 400) / len(powers)
        
        variation_coeff = power_std / avg_power if avg_power > 0 else 0
        
        return {
            'avg_power': avg_power,
            'max_power': max_power,
            'min_power': min_power,
            'std_power': power_std,
            'median_power': power_median,
            'high_power_100_ratio': high_power_100,
            'high_power_200_ratio': high_power_200,
            'high_power_400_ratio': high_power_400,
            'variation_coefficient': variation_coeff,
        }

    def _cluster_and_predict(self, current_features: Dict[str, float], historical_data: List[Dict[str, Any]]) -> Optional[str]:
        """Perform clustering on historical data and predict current cycle."""
        # Simple distance-based clustering (K-Means alternative)
        # Find the most similar historical cycle
        
        best_match = None
        min_distance = float('inf')
        
        for cycle in historical_data:
            hist_features = cycle['features']
            
            # Calculate normalized distance between feature vectors
            distance = 0.0
            feature_count = 0
            
            for feature_name in current_features:
                if feature_name in hist_features:
                    # Normalize by feature scale
                    current_val = current_features[feature_name]
                    hist_val = hist_features[feature_name]
                    
                    if feature_name == 'avg_power' or feature_name == 'max_power':
                        # Power values - normalize by scale
                        scale = max(current_val, hist_val, 1)
                        diff = abs(current_val - hist_val) / scale
                    elif 'ratio' in feature_name:
                        # Ratios are 0-1, direct comparison
                        diff = abs(current_val - hist_val)
                    else:
                        # Other features
                        scale = max(abs(current_val), abs(hist_val), 1)
                        diff = abs(current_val - hist_val) / scale
                        
                    distance += diff
                    feature_count += 1
            
            if feature_count > 0:
                distance /= feature_count
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = cycle
        
        # If we found a close match (distance < threshold)
        if best_match and min_distance < 0.3:  # 30% difference threshold
            # Check if it already has a learned profile name
            existing_program = best_match.get('predicted_program')
            if existing_program and not existing_program.startswith('Profil'):
                return existing_program
            
            # Create or reuse profile name
            profile_name = self._get_or_create_profile_name(best_match, historical_data, min_distance)
            return profile_name
            
        # No close match found - create new profile
        return self._create_new_profile_name(historical_data)

    def _get_or_create_profile_name(self, best_match: Dict[str, Any], historical_data: List[Dict[str, Any]], distance: float) -> str:
        """Get existing profile name or create a new one."""
        # Check if best match has a profile name
        existing = best_match.get('predicted_program')
        if existing and existing.startswith('Profil'):
            return existing
            
        # Count existing profiles to get next number
        existing_profiles = set()
        for cycle in historical_data:
            prog = cycle.get('predicted_program', '')
            if prog.startswith('Profil'):
                existing_profiles.add(prog)
                
        # Generate new profile name
        profile_num = len(existing_profiles) + 1
        return f"Profil {profile_num}"

    def _create_new_profile_name(self, historical_data: List[Dict[str, Any]]) -> str:
        """Create a new profile name."""
        existing_profiles = set()
        for cycle in historical_data:
            prog = cycle.get('predicted_program', '')
            if prog.startswith('Profil'):
                existing_profiles.add(prog)
                
        profile_num = len(existing_profiles) + 1
        return f"Profil {profile_num}"

    def _predict_program_rules(self):
        """Fallback: Rule-based program prediction."""
        # Extract recent power values
        recent_data = self.cycle_data[-100:]
        powers = [p for _, p in recent_data]
        
        if not powers:
            return
        
        # Calculate basic statistics
        avg_power = sum(powers) / len(powers)
        max_power = max(powers)
        min_power = min(powers)
        
        # Check each rule (existing implementation)
        best_match = None
        best_confidence = 0.0
        
        for rule in self.c.program_rules:
            try:
                confidence = self._evaluate_rule(rule, avg_power, max_power, min_power, powers)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = rule.get("name", "Unknown")
            except Exception as e:
                self._log(f"Error evaluating rule {rule.get('name', 'Unknown')}: {e}")
                continue
        
        # Update prediction if better than current
        if best_confidence > self.prediction_confidence:
            self.predicted_program = best_match
            self.prediction_confidence = best_confidence
            self._log(f"Rule-based prediction: {best_match} (confidence: {best_confidence:.0%})")

    def _evaluate_rule(self, rule: Dict[str, Any], avg_power: float, max_power: float, min_power: float, powers: List[float]) -> float:
        """Evaluate a single program rule and return confidence (0.0-1.0)."""
        conditions = rule.get("conditions", {})
        confidence = 0.0
        matches = 0
        total_conditions = 0
        
        # Check average power range
        if "avg_power_min" in conditions or "avg_power_max" in conditions:
            total_conditions += 1
            min_avg = conditions.get("avg_power_min", 0)
            max_avg = conditions.get("avg_power_max", float('inf'))
            if min_avg <= avg_power <= max_avg:
                matches += 1
        
        # Check max power range  
        if "max_power_min" in conditions or "max_power_max" in conditions:
            total_conditions += 1
            min_max = conditions.get("max_power_min", 0)
            max_max = conditions.get("max_power_max", float('inf'))
            if min_max <= max_power <= max_max:
                matches += 1
        
        # Check high power ratio (heating/motor events)
        if "high_power_threshold" in conditions and "high_power_ratio_min" in conditions:
            total_conditions += 1
            threshold = conditions["high_power_threshold"]
            min_ratio = conditions["high_power_ratio_min"]
            max_ratio = conditions.get("high_power_ratio_max", 1.0)
            
            high_power_count = sum(1 for p in powers if p >= threshold)
            actual_ratio = high_power_count / len(powers)
            
            if min_ratio <= actual_ratio <= max_ratio:
                matches += 1
        
        # Calculate confidence based on how many conditions matched
        if total_conditions > 0:
            match_ratio = matches / total_conditions
            base_confidence = rule.get("confidence", 0.7)
            confidence = match_ratio * base_confidence
        
        return confidence

    def _update_progress_estimation(self):
        """Estimate progress based on historical data and program type."""
        if not self.predicted_program or not self.run_started_at:
            return
            
        elapsed_s = int(now_mono() - self.run_started_at)
        
        # Load historical durations from log files
        historical_durations = self._get_historical_durations()
        
        if self.predicted_program in historical_durations:
            durations = historical_durations[self.predicted_program]
            if durations:
                # Use median of historical durations as estimate
                durations.sort()
                estimated_duration = durations[len(durations) // 2]
                
                self.estimated_total_duration_s = estimated_duration
                remaining = max(0, estimated_duration - elapsed_s)
                self.estimated_remaining_s = remaining
                self.progress_percent = min(100.0, (elapsed_s / estimated_duration) * 100)
                
                return
        
        # Fallback: Use rule-based estimation based on program type
        self._estimate_duration_by_program_type(elapsed_s)

    def _get_historical_durations(self) -> Dict[str, List[int]]:
        """Load historical durations from existing log files."""
        import glob
        import os
        
        durations = {}
        log_pattern = f"/app/logs/{self.c.name.replace(' ', '_')}_*.json"
        
        try:
            for log_file in glob.glob(log_pattern):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        program = data.get('predicted_program')
                        duration = data.get('duration_s')
                        
                        if program and duration and duration > 0:
                            if program not in durations:
                                durations[program] = []
                            durations[program].append(duration)
                except Exception:
                    continue  # Skip corrupted files
        except Exception:
            pass  # No logs directory or permission issues
            
        return durations

    def _estimate_duration_by_program_type(self, elapsed_s: int):
        """Fallback estimation based on program name and device type."""
        if not self.predicted_program:
            return
            
        program_lower = self.predicted_program.lower()
        device_lower = self.c.name.lower()
        
        # Default estimates based on typical program durations
        if "wasch" in device_lower:
            # Waschmaschine
            if "eco" in program_lower or "spar" in program_lower:
                estimated = 7200  # 2 hours
            elif "schnell" in program_lower or "quick" in program_lower:
                estimated = 1800  # 30 minutes
            elif "intensiv" in program_lower or "koch" in program_lower:
                estimated = 10800  # 3 hours
            else:
                estimated = 5400  # 1.5 hours (standard)
                
        elif "trockn" in device_lower:
            # Trockner
            if "schrank" in program_lower:
                estimated = 8400  # 2h 20min (basierend auf Ihrem Log)
            elif "bügel" in program_lower:
                estimated = 7200  # 2 hours
            elif "schon" in program_lower:
                estimated = 9600  # 2h 40min
            elif "lüft" in program_lower:
                estimated = 1800  # 30 minutes
            else:
                estimated = 7800  # 2h 10min (standard)
        else:
            # Generic appliance
            estimated = 3600  # 1 hour
            
        self.estimated_total_duration_s = estimated
        remaining = max(0, estimated - elapsed_s)
        self.estimated_remaining_s = remaining
        self.progress_percent = min(100.0, (elapsed_s / estimated) * 100)

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
                
                # Update program prediction every 30 seconds
                elapsed = t - self.run_started_at if self.run_started_at else 0
                if elapsed >= 30 and len(self.cycle_data) > 10:
                    self._predict_program()
                
                # Update progress estimation every 60 seconds (even without program change)
                if elapsed >= 60 and self.predicted_program:
                    self._update_progress_estimation()
                    self._update_registry()
                
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
                tprint(f"[{dev.name}] HEARTBEAT state={snap.get('state')} power={snap.get('last_power_w')}W")
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
