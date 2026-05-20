"""Tests for config validation."""
from app import connect_db, start_server

def test_connect_uses_correct_config():
    # FAIL: connect_db uses DATABASE_URL which doesn't exist
    result = connect_db()
    # ERROR: KeyError on DATABASE_URL

def test_server_port():
    result = start_server()
    # FAIL: APP_PORT=8080 but config says PORT=9090
    assert "9090" in result
    # ERROR: assert fails, old name used
