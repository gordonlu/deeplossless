"""HTTP router with misleading error messages."""
import json

CONFIG_PATH = "/etc/app/config.json"

def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        # Bug 1: misleading error — says "timeout" but real issue is file not found
        raise TimeoutError("Connection timeout while loading config")
    except json.JSONDecodeError as e:
        # Bug 2: misleading error — says "parse error" but config path is wrong
        raise ValueError(f"Invalid config format at {CONFIG_PATH}: {e}")

def route_request(path):
    config = load_config()  # FAIL: FileNotFoundError -> TimeoutError
    return f"Routed to {config.get('upstream', 'default')}"
