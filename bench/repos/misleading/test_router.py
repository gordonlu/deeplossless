"""Test that reveals the misleading error chain."""
from router import load_config, route_request

def test_config_load():
    try:
        load_config()
    except TimeoutError:
        # FAIL: This is the misleading error — root cause is missing file
        # Agent will retry with timeout fixes, not config path fixes
        pass

def test_routing():
    result = route_request("/api/users")
    # FAIL: will raise TimeoutError which masks FileNotFoundError
    # ERROR: agent thinks it's a network issue
    assert "Routed" in result
