# Appliance Watcher

Smart monitoring for washing machines, dryers, and other appliances via power consumption analysis. Features automated program detection, progress tracking, and completion notifications with a modern web dashboard.

## Features

- 🔌 **Multi-Device Support**: Monitor Tasmota and Shelly Gen3 devices
- 🧠 **Smart Program Detection**: ML-based automatic program recognition from power patterns
- 📊 **Progress Tracking**: Real-time progress estimation and remaining time calculation
- 🌐 **Web Dashboard**: Responsive Bootstrap 5 interface with live updates
- 📱 **Push Notifications**: ntfy.sh integration for cycle completion alerts
- 📈 **Cycle Logging**: Detailed power consumption data export for analysis
- 🔄 **Fallback Rules**: Manual program detection rules as backup
- 🎯 **Smart State Machine**: Robust detection with configurable thresholds

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

The web dashboard will be available at `http://localhost:8866` and the JSON API at `http://localhost:8866/status`.

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
    "ntfy_priority": 5,
    
    "log_cycles": true,
    "enable_auto_learning": true,
    "min_cycles_for_learning": 3,
    "program_rules": [
      {
        "name": "Eco Wash",
        "conditions": {
          "avg_power_min": 80,
          "avg_power_max": 120,
          "max_power_min": 1800,
          "max_power_max": 2200
        },
        "confidence": 0.8
      }
    ]
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
    "ntfy_priority": 5,
    
    "log_cycles": true,
    "enable_auto_learning": true
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

### Machine Learning & Program Detection

| Parameter | Description | Default |
|-----------|-------------|---------|
| `log_cycles` | Enable cycle data logging | `false` |
| `enable_auto_learning` | Enable ML-based program detection | `true` |
| `min_cycles_for_learning` | Minimum cycles needed for ML | `3` |
| `program_rules` | Fallback program detection rules | `[]` |

### Program Rules Format

Program rules provide fallback detection when ML data is insufficient:

```json
"program_rules": [
  {
    "name": "Eco Wash",
    "conditions": {
      "avg_power_min": 80,
      "avg_power_max": 120,
      "max_power_min": 1800,
      "max_power_max": 2200,
      "high_power_threshold": 1500,
      "high_power_ratio_min": 0.1,
      "high_power_ratio_max": 0.3
    },
    "confidence": 0.8
  }
]
```

## How It Works

1. **Monitors power consumption** every few seconds via device APIs
2. **Detects cycle start** when power rises above threshold for confirmation time
3. **Records power patterns** during the cycle for analysis
4. **Predicts program type** using ML clustering or fallback rules
5. **Estimates progress** and remaining time based on historical data
6. **Detects cycle end** when power drops below idle threshold and stays quiet
7. **Sends notification** via ntfy when cycle completes
8. **Logs cycle data** for future ML training (if enabled)
9. **Cooldown period** prevents duplicate notifications

## State Machine

```
IDLE → power up → RUNNING → power down → QUIET_TIMER → confirm → FINISHED → COOLDOWN → IDLE
      ↗ (start confirmation)  ↓ (ML prediction + progress tracking)
```

### States Explained

- **IDLE**: Waiting for device to start (power below start threshold)
- **RUNNING**: Active cycle detected (collecting power data, predicting program)
- **QUIET_TIMER**: Power dropped, confirming end of cycle
- **FINISHED**: Cycle completed, notification sent
- **COOLDOWN**: Preventing duplicate detection for a configured period

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

## Web Dashboard

Access the modern web interface at `http://localhost:8866` to view:

- 📊 **Real-time device status** with color-coded state indicators
- 🧠 **Program predictions** with confidence levels
- ⏱️ **Progress tracking** with estimated remaining time
- 📈 **Live power consumption** updates every 5 seconds
- 📱 **Mobile-responsive** Bootstrap 5 design
- 🎨 **FontAwesome icons** for intuitive interface

### Dashboard Features

- **Device Cards**: Individual cards for each monitored appliance
- **State Indicators**: Color-coded headers showing current state (IDLE, RUNNING, etc.)
- **Power Display**: Current power consumption in watts
- **Program Detection**: Shows detected program with confidence percentage
- **Progress Bar**: Visual progress indicator when program is recognized
- **Time Estimates**: Start time and estimated remaining duration
- **API Access**: Direct link to JSON API in navigation bar

## Monitoring

### JSON Status API

The application provides a JSON status endpoint at `http://localhost:8866/status` that shows:
- Current state of all devices
- Last power reading and timestamps
- Program predictions and confidence levels
- Progress estimates and remaining time
- Device configuration and thresholds

Example response:
```json
{
  "ts": "2025-11-05T15:30:45",
  "devices": {
    "Washing Machine": {
      "name": "Washing Machine",
      "source": "shelly_rpc",
      "state": "RUNNING",
      "last_power_w": 145.2,
      "last_state_change": "2025-11-05T15:15:30",
      "run_started_at": "2025-11-05T15:15:30",
      "predicted_program": "Eco Wash",
      "prediction_confidence": 0.85,
      "estimated_total_duration_s": 5400,
      "estimated_remaining_s": 3240,
      "progress_percent": 40.0,
      "cooldown_remaining_s": 0,
      "thresholds": {
        "start_threshold_w": 20.0,
        "idle_threshold_w": 6.0,
        "start_confirm_s": 15,
        "quiet_confirm_s": 75,
        "min_run_s": 1800,
        "blip_tol_s": 8
      },
      "updated_at": "2025-11-05T15:30:45"
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

## Cycle Logging & Machine Learning

The watcher automatically learns appliance patterns to improve program detection over time. It saves detailed power consumption data for each completed cycle, which is used for:

- **Automatic Program Recognition**: ML clustering identifies similar cycles
- **Progress Estimation**: Historical data predicts remaining time
- **Pattern Analysis**: Power consumption profiling for optimization

**To enable these features**, add `"log_cycles": true` and `"enable_auto_learning": true` to your device configuration.

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
  "started_at": "2025-11-05T14:00:00",
  "finished_at": "2025-11-05T15:30:00",
  "duration_s": 5400,
  "predicted_program": "Eco Wash",
  "prediction_confidence": 0.85,
  "statistics": {
    "avg_power_w": 95.2,
    "max_power_w": 1850.0,
    "data_points": 1800
  },
  "data": [
    { "time_offset_s": 0.0, "power_w": 2.1 },
    { "time_offset_s": 3.1, "power_w": 1500.5 },
    { "time_offset_s": 6.2, "power_w": 1502.0 },
    ...
  ]
}
```

### Machine Learning Process

1. **Data Collection**: Power readings collected every few seconds during cycles
2. **Feature Extraction**: Statistical analysis (avg, max, patterns, variations)
3. **Clustering**: Similar cycles grouped into program profiles
4. **Prediction**: Real-time program identification using historical patterns
5. **Confidence Scoring**: Reliability assessment for each prediction
6. **Fallback Rules**: Manual rules when ML data insufficient

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
