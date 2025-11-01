# Appliance Watcher

Monitor washing machines, dryers, and other appliances via power consumption. Get notified when cycles complete.

## Supported Devices

- **Tasmota** devices (via HTTP API)
- **Shelly Gen3** devices (via RPC API)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd appliance-watcher
   ```

2. **Configure devices**
   ```bash
   cp devices.json.example devices.json
   # Edit devices.json with your device IPs and settings
   ```

3. **Set credentials** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your ntfy credentials if needed
   ```

4. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

The service will be available at `http://localhost:8866/status` for monitoring.

## Configuration

### 1. Environment Variables (`.env`)

```properties
NTFY_USER=your_ntfy_username
NTFY_PASS=your_ntfy_password
```

Optional monitoring settings:
```properties
STATUS_HTTP=0.0.0.0:8080
HEARTBEAT_SECONDS=60
```

### 2. Device Configuration (`devices.json`)

Example for Shelly device:

```json
[
  {
    "name": "Washing Machine",
    "source": "shelly_rpc",
    "base_url": "http://192.168.1.100",
    "device_user": null,
    "device_pass": null,
    "shelly_rpc_method": "PM1.GetStatus",
    "shelly_rpc_params": { "id": 0 },
    "shelly_power_field": "apower",
    
    "poll_interval": 3,
    "start_threshold_w": 20,
    "idle_threshold_w": 6,
    "start_confirm_seconds": 15,
    "quiet_confirm_seconds": 75,
    "min_run_seconds": 1800,
    "blip_tolerance_seconds": 8,
    "cooldown_seconds": 150,
    
    "ntfy_url": "https://ntfy.sh",
    "ntfy_topic": "appliances",
    "ntfy_title": "Washing Machine",
    "ntfy_icon": "https://example.com/washer.png",
    "ntfy_priority": 5
  }
]
```

Example for Tasmota device:

```json
[
  {
    "name": "Dryer",
    "source": "tasmota",
    "base_url": "http://192.168.1.101",
    "device_user": null,
    "device_pass": null,
    
    "poll_interval": 3,
    "start_threshold_w": 25,
    "idle_threshold_w": 5,
    "start_confirm_seconds": 12,
    "quiet_confirm_seconds": 60,
    "min_run_seconds": 3600,
    "blip_tolerance_seconds": 6,
    "cooldown_seconds": 180,
    
    "ntfy_url": "https://ntfy.sh",
    "ntfy_topic": "appliances",
    "ntfy_title": "Dryer",
    "ntfy_priority": 5
  }
]
```

## Configuration Parameters

### Device Settings

| Parameter | Description | Example |
|-----------|-------------|---------|
| `name` | Device display name | `"Washing Machine"` |
| `source` | Device type: `"tasmota"` or `"shelly_rpc"` | `"shelly_rpc"` |
| `base_url` | Device IP/URL | `"http://192.168.1.100"` |
| `device_user` | Basic auth username (optional) | `"admin"` |
| `device_pass` | Basic auth password (optional) | `"password"` |

### Shelly RPC Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `shelly_rpc_method` | RPC method to call | `"PM1.GetStatus"` |
| `shelly_rpc_params` | RPC parameters | `{"id": 0}` |
| `shelly_power_field` | Power field name in response | `"apower"` |

### Detection Logic

| Parameter | Description | Default |
|-----------|-------------|---------|
| `poll_interval` | Seconds between power checks | `3` |
| `start_threshold_w` | Power level to detect start | `10.0` |
| `idle_threshold_w` | Power level for idle detection | `3.0` |
| `start_confirm_seconds` | Confirm start after X seconds | `12` |
| `quiet_confirm_seconds` | Confirm end after X seconds quiet | `60` |
| `min_run_seconds` | Minimum cycle duration | `180` |
| `blip_tolerance_seconds` | Ignore brief power drops | `6` |
| `cooldown_seconds` | Wait before detecting new cycle | `150` |

### Notifications

| Parameter | Description | Example |
|-----------|-------------|---------|
| `ntfy_url` | ntfy server URL | `"https://ntfy.sh"` |
| `ntfy_topic` | ntfy topic | `"appliances"` |
| `ntfy_title` | Notification title | `"Washing Machine"` |
| `ntfy_icon` | Icon URL (optional) | `"https://example.com/icon.png"` |
| `ntfy_priority` | Priority level (1-5) | `5` |

## How It Works

1. **Monitors power consumption** every few seconds
2. **Detects cycle start** when power rises above threshold for confirmation time
3. **Detects cycle end** when power drops below idle threshold and stays quiet
4. **Sends notification** via ntfy when cycle completes
5. **Cooldown period** prevents duplicate notifications

## State Machine

```
IDLE → power up → RUNNING → power down → QUIET_TIMER → confirm → FINISHED → COOLDOWN → IDLE
```

## Docker Compose

The included `docker-compose.yml` automatically:
- Builds the container
- Mounts `devices.json` as read-only
- Loads environment variables from `.env`
- Exposes status server on port 8866
- Restarts on failure

## Logs

View logs with:

```bash
docker-compose logs -f appliance-watch
```

## Monitoring

### Status Server

The application provides a JSON status endpoint at `http://localhost:8866/status` that shows:
- Current state of all devices
- Last power reading
- Timestamps of state changes

Example response:
```json
{
  "ts": "2025-10-27T15:30:45",
  "devices": {
    "Washing Machine": {
      "state": "RUNNING",
      "last_power_w": 45.2,
      "run_started_at": "2025-10-27T15:15:30"
    }
  }
}
```

### Heartbeat Logs

Set `HEARTBEAT_SECONDS=60` to get periodic status updates in logs every 60 seconds.

## Troubleshooting

### Device not responding
- Check device IP address and network connectivity
- Verify device credentials if using authentication
- Test device API manually:
  - Tasmota: `curl http://IP/cm?cmnd=Status%2010`
  - Shelly: `curl http://IP/rpc/PM1.GetStatus?id=0`

### No notifications
- Verify ntfy credentials in `.env`
- Check ntfy topic and URL
- Test ntfy manually: `curl -d "test" https://ntfy.sh/your-topic`

### False triggers
- Adjust `start_threshold_w` and `idle_threshold_w`
- Increase `start_confirm_seconds` and `quiet_confirm_seconds`
- Check power consumption patterns in logs
- Monitor device states via status endpoint

### Monitoring Health
- Check status endpoint: `curl http://localhost:8866/status`
- Enable heartbeat logs with `HEARTBEAT_SECONDS=60`
- Monitor container health: `docker-compose ps`

## Cycle Logging

For detailed analysis and profiling of appliance cycles (e.g., washing programs), the watcher can save a complete power profile for each completed run.

When a device cycle finishes, the application will create a new JSON file in the `/app/logs/` directory inside the container.

**To enable this feature**, you must add `"log_cycles": true` to the device's configuration in your `devices.json` file.

When a device cycle finishes (and `log_cycles` is enabled), the application will create a new JSON file in the `/app/logs/` directory inside the container.

### File Naming and Content

-   **Location**: `/app/logs/`
-   **Filename**: `<Device-Name>_<Timestamp>.json` (e.g., `Washing_Machine_2023-10-27_15-30-00.json`)
-   **Content**: Each file is a single JSON object containing:
    -   Device name, start/finish times, and total duration.
    -   A `data` array with power readings (`power_w`) and their time offset in seconds from the start of the cycle.

**Example JSON file:**
```json
{
  "device": "Washing Machine",
  "started_at": "2023-10-27T14:00:00",
  "finished_at": "2023-10-27T15:30:00",
  "duration_s": 5400,
  "data": [
    { "time_offset_s": 0.0, "power_w": 2.1 },
    { "time_offset_s": 3.1, "power_w": 1500.5 },
    { "time_offset_s": 6.2, "power_w": 1502.0 },
    ...
  ]
}
```

### Accessing the Log Files

To access these log files on your host machine, you must map a local directory to the `/app/logs` directory in the container. Add the following volume mapping to your `docker-compose.yml`:

```yaml
services:
  tasmota_watcher:
    # ... other config ...
    volumes:
      - ./logs:/app/logs
      - ./devices.json:/app/devices.json:ro
```

This will cause all log files to appear in a `logs` folder in your project directory.
