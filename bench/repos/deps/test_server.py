"""Tests that will fail due to dependency bugs."""
from server import Server

def test_server_init():
    s = Server()
    assert s is not None
    # FAIL: This should work but Database init crashes
    # ERROR: old_db_driver.Database requires sync context

def test_health_check():
    s = Server()
    # FAIL: TimeoutError from deprecated async-timeout
    import asyncio
    result = asyncio.run(s.health_check())
    assert result["status"] == "ok"
    # ERROR: cannot call sync Database in async health_check
