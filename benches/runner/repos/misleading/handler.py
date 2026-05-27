"""Request handler — propagates misleading errors."""
from router import load_config

def handle(path):
    try:
        config = load_config()
        return f"Handled {path} with {config}"
    except TimeoutError as e:
        # Agent sees "TimeoutError" and tries to fix timeout
        # But the REAL bug is missing config file
        return f"Error: timeout — check network"  # WRONG FIX SUGGESTION
    except Exception as e:
        return f"Error: {e}"
